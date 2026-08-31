"""从同一 LoRA 结构创建函数为零的战略与战斗 residual。"""

import json
import math
import re
import shutil
from pathlib import Path

from ..learner import GrpoTrainingError


def initialize_residual_adapters(
    *,
    template_adapter: Path,
    output_root: Path,
    strategy_name: str,
    battle_name: str,
    seed: int,
) -> dict[str, str]:
    """创建两个标准 ``A`` 随机、``B`` 为零的 residual LoRA。

    只复用模板的结构和目标模块，不复制模板已经学到的 LoRA 函数。两个 adapter
    在创建时都对冻结 merged-SFT base 产生严格零增量，随后由不同数据源更新。

    Args:
        template_adapter (Path): 提供 LoRA 结构的已发布 adapter。
        output_root (Path): 两个 residual 的输出父目录。
        strategy_name (str): 战略 residual 的可读名称。
        battle_name (str): 战斗 residual 的可读名称。
        seed (int): 战略 adapter 的随机初始化种子；战斗使用 ``seed + 1``。

    Raises:
        GrpoTrainingError: 模板、名称、权重或输出目录无效。

    Returns:
        dict[str, str]: 两个新 adapter 的路径。
    """
    template = Path(template_adapter)
    config_path = template / "adapter_config.json"
    weights_path = template / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise GrpoTrainingError("residual 模板 adapter 不完整")
    names = (strategy_name, battle_name)
    if strategy_name == battle_name or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) for name in names
    ):
        raise GrpoTrainingError("双 residual 名称必须不同且可读")
    destinations = tuple(Path(output_root) / name for name in names)
    if any(destination.exists() for destination in destinations):
        raise GrpoTrainingError("双 residual 输出已存在")

    import torch
    from safetensors.torch import load_file, save_file

    template_state = load_file(str(weights_path), device="cpu")
    if not template_state:
        raise GrpoTrainingError("residual 模板没有权重")
    for index, destination in enumerate(destinations):
        generator_seed = seed + index
        torch.manual_seed(generator_seed)
        state = {}
        found_a = False
        found_b = False
        for name, tensor in template_state.items():
            value = torch.empty_like(tensor)
            if "lora_A" in name:
                torch.nn.init.kaiming_uniform_(value, a=math.sqrt(5))
                found_a = True
            elif "lora_B" in name:
                value.zero_()
                found_b = True
            else:
                raise GrpoTrainingError(f"residual 模板含非 LoRA 权重: {name}")
            state[name] = value
        if not found_a or not found_b:
            raise GrpoTrainingError("residual 模板缺少 LoRA A/B 权重")
        destination.mkdir(parents=True)
        shutil.copy2(config_path, destination / "adapter_config.json")
        save_file(state, str(destination / "adapter_model.safetensors"))
        (destination / "residual_manifest.json").write_text(
            json.dumps(
                {
                    "name": names[index],
                    "template_adapter": str(template),
                    "initialization": "kaiming_A_zero_B",
                    "seed": generator_seed,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "strategy_adapter": str(destinations[0]),
        "battle_adapter": str(destinations[1]),
    }


def compose_serving_adapter(
    *,
    parent_adapter: Path,
    residual_adapter: Path,
    output: Path,
    name: str,
) -> dict[str, str | int]:
    """把 SFT LoRA 与一个 residual 精确拼成单个 vLLM serving LoRA。

    训练仍在函数等价的 merged-SFT base 上只更新 residual；vLLM 请求一次只能激活
    一个 LoRA，因此服务产物按 rank 维拼接两个低秩分解。输出采用
    ``alpha = combined_rank``，并把各输入 ``alpha/r`` 吸收到对应 B 矩阵，函数为
    ``Delta_parent + Delta_residual``，不做平均或近似 merge。

    Args:
        parent_adapter (Path): 冻结最终 SFT LoRA。
        residual_adapter (Path): 当前战略或战斗 residual。
        output (Path): 尚不存在的 composite adapter 目录。
        name (str): 可读 serving policy 名。

    Raises:
        GrpoTrainingError: 配置、键、shape、特性或输出无效。

    Returns:
        dict[str, str | int]: 输出路径、名称与组合 rank。
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise GrpoTrainingError("composite serving policy 名称无效")
    destination = Path(output)
    if destination.exists():
        raise GrpoTrainingError("composite serving adapter 输出已存在")
    parent_config = _lora_config(Path(parent_adapter))
    residual_config = _lora_config(Path(residual_adapter))
    scalar_fields = ("fan_in_fan_out", "bias", "task_type")
    parent_targets = parent_config.get("target_modules")
    residual_targets = residual_config.get("target_modules")
    if (
        not isinstance(parent_targets, list)
        or not isinstance(residual_targets, list)
        or set(parent_targets) != set(residual_targets)
        or any(
            parent_config.get(field) != residual_config.get(field)
            for field in scalar_fields
        )
    ):
        raise GrpoTrainingError("parent 与 residual LoRA 结构不一致")
    for config in (parent_config, residual_config):
        if (
            config.get("use_rslora") is True
            or config.get("use_dora") is True
            or float(config.get("lora_dropout", 0.0)) != 0.0
            or config.get("rank_pattern") not in (None, {})
            or config.get("alpha_pattern") not in (None, {})
        ):
            raise GrpoTrainingError("composite serving 只支持普通零 dropout LoRA")
    try:
        parent_rank = int(parent_config["r"])
        residual_rank = int(residual_config["r"])
        parent_scale = float(parent_config["lora_alpha"]) / parent_rank
        residual_scale = float(residual_config["lora_alpha"]) / residual_rank
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise GrpoTrainingError("composite serving LoRA rank/alpha 无效") from exc
    if parent_rank <= 0 or residual_rank <= 0:
        raise GrpoTrainingError("composite serving LoRA rank 必须为正")

    from safetensors.torch import load_file, save_file

    parent_state = load_file(
        str(Path(parent_adapter) / "adapter_model.safetensors"),
        device="cpu",
    )
    residual_state = load_file(
        str(Path(residual_adapter) / "adapter_model.safetensors"),
        device="cpu",
    )
    if set(parent_state) != set(residual_state):
        raise GrpoTrainingError("parent 与 residual LoRA 权重键不一致")
    combined = {}
    for key in sorted(parent_state):
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in parent_state:
            raise GrpoTrainingError(f"composite serving 缺少 B 矩阵: {key}")
        parent_a = parent_state[key]
        residual_a = residual_state[key]
        parent_b = parent_state[b_key]
        residual_b = residual_state[b_key]
        if (
            parent_a.shape[1:] != residual_a.shape[1:]
            or parent_b.shape[:-1] != residual_b.shape[:-1]
            or parent_a.shape[0] != parent_rank
            or residual_a.shape[0] != residual_rank
            or parent_b.shape[-1] != parent_rank
            or residual_b.shape[-1] != residual_rank
        ):
            raise GrpoTrainingError(f"composite serving LoRA shape 不一致: {key}")
        combined[key] = __import__("torch").cat((parent_a, residual_a), dim=0)
        combined[b_key] = __import__("torch").cat(
            (parent_b * parent_scale, residual_b * residual_scale),
            dim=-1,
        )
    if not combined or len(combined) != len(parent_state):
        raise GrpoTrainingError("composite serving LoRA 含非 A/B 权重")
    combined_rank = parent_rank + residual_rank
    config = dict(parent_config)
    config["r"] = combined_rank
    config["lora_alpha"] = combined_rank
    config["inference_mode"] = True
    destination.mkdir(parents=True)
    (destination / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_file(combined, str(destination / "adapter_model.safetensors"))
    (destination / "serving_manifest.json").write_text(
        json.dumps(
            {
                "name": name,
                "parent_adapter": str(parent_adapter),
                "residual_adapter": str(residual_adapter),
                "composition": "rank_concat_exact_sum",
                "rank": combined_rank,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"name": name, "output": str(destination), "rank": combined_rank}


def _lora_config(root: Path) -> dict[str, object]:
    """读取一个 LoRA adapter 配置对象。

    Args:
        root (Path): adapter 目录。

    Raises:
        GrpoTrainingError: 配置文件缺失或不是 JSON 对象。

    Returns:
        dict[str, object]: 可供结构比较的配置。
    """
    path = Path(root) / "adapter_config.json"
    weights = Path(root) / "adapter_model.safetensors"
    if not path.is_file() or not weights.is_file():
        raise GrpoTrainingError(f"LoRA adapter 不完整: {root}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GrpoTrainingError(f"LoRA adapter 配置无效: {root}") from exc
    if not isinstance(payload, dict):
        raise GrpoTrainingError(f"LoRA adapter 配置必须是对象: {root}")
    return payload
