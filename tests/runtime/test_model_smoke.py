"""验证模型服务与真实 Harness 之间的最小协议冒烟。"""

import hashlib
import json
from pathlib import Path

import httpx
import pytest


def test_smoke_model_verifies_running_artifact_and_generates_legal_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """冒烟检查先访问健康端点，再用真实 Harness 契约生成合法动作。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实磁盘身份检查。

    Raises:
        AssertionError: 请求顺序、提示词内容或生成参数不符合约定。

    Returns:
        None: 此测试只验证 OpenAI-compatible 服务协议。
    """
    from play_sts2.runtime import model_smoke

    merged_model, serving_model = _model_fixture(tmp_path)
    disk_checks: list[tuple[Path, str, Path]] = []

    def validate_disk(
        model_dir: Path,
        *,
        artifact_id: str,
        merged_model: Path,
    ) -> None:
        """记录冒烟前完成的服务制品磁盘身份检查。

        Args:
            model_dir (Path): 待启动的 MLX 服务模型目录。
            artifact_id (str): 期望的训练制品标识。
            merged_model (Path): 服务制品声明的合并模型目录。

        Returns:
            None: 此替身只记录磁盘身份检查参数。
        """
        disk_checks.append((model_dir, artifact_id, merged_model))

    monkeypatch.setattr(model_smoke, "validate_serving_model", validate_disk)
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """返回 MLX 健康响应和合法 chat completion。

        Args:
            request (httpx.Request): 本地模型检查发出的 HTTP 请求。

        Raises:
            AssertionError: 请求不是健康检查或 Harness 生成请求。

        Returns:
            httpx.Response: 对应端点的生产等价响应。
        """
        paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": str(serving_model.resolve()), "object": "model"}],
                },
            )
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == str(serving_model.resolve())
        if body["max_tokens"] == 1:
            assert body["chat_template_kwargs"] == {"enable_thinking": True}
            return httpx.Response(
                200,
                json={
                    "model": str(serving_model.resolve()),
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                            },
                            "finish_reason": "length",
                        }
                    ],
                },
            )
        assert body["max_tokens"] == 512
        assert body["temperature"] == 0.2
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        system_message = body["messages"][0]["content"]
        assert "战斗决策模型" in system_message
        user_message = body["messages"][1]["content"]
        assert "可执行动作:\n- end_turn" in user_message
        if "ACTION 绝不能写在思考区内" not in user_message:
            return httpx.Response(
                200,
                json={
                    "model": str(serving_model.resolve()),
                    "choices": [
                        {
                            "message": {
                                "reasoning": "ACTION: end_turn",
                                "content": "",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "model": str(serving_model.resolve()),
                "choices": [
                    {
                        "message": {
                            "reasoning": "唯一动作是结束回合",
                            "content": "ACTION: end_turn",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    reply = model_smoke.smoke_model(
        "http://127.0.0.1:8900",
        artifact_id="demo-e3",
        merged_model=merged_model,
        serving_model=serving_model,
        enable_thinking=True,
        max_tokens=512,
        temperature=0.2,
        transport=httpx.MockTransport(respond),
    )

    assert paths == [
        "/health",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert reply.text == "ACTION: end_turn"
    assert reply.reasoning == "唯一动作是结束回合"
    assert disk_checks == [(serving_model, "demo-e3", merged_model)]


def test_smoke_model_rejects_different_running_model_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """磁盘身份正确但端口仍运行旧模型时，冒烟必须拒绝假绿。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实磁盘身份检查。

    Raises:
        AssertionError: 身份不符时仍发起了生成请求。

    Returns:
        None: 此测试只检查运行中模型身份门禁。
    """
    from play_sts2.inference import ServingModelIdentityError
    from play_sts2.runtime import model_smoke

    merged_model, serving_model = _model_fixture(tmp_path)
    monkeypatch.setattr(
        model_smoke,
        "validate_serving_model",
        lambda *_args, **_kwargs: None,
    )
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """声明端口仍加载旧模型并拒绝后续生成。

        Args:
            request (httpx.Request): 冒烟流程发出的 HTTP 请求。

        Raises:
            AssertionError: 身份检查失败后仍尝试调用生成端点。

        Returns:
            httpx.Response: 健康端点或旧模型身份响应。
        """
        paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [{"id": "/models/serving/demo-e2-mlx-8bit"}],
                },
            )
        raise AssertionError("模型身份不符时不应请求生成")

    with pytest.raises(ServingModelIdentityError, match="运行中的模型目录不一致"):
        model_smoke.smoke_model(
            "http://127.0.0.1:8900",
            artifact_id="demo-e3",
            merged_model=merged_model,
            serving_model=serving_model,
            enable_thinking=False,
            max_tokens=128,
            temperature=0.0,
            transport=httpx.MockTransport(respond),
        )

    assert paths == ["/health", "/v1/models"]


def test_validate_model_service_rejects_model_switch_during_readiness_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """声明 e3 后生成响应却来自 e2 时，必须在接触游戏前失败。

    Args:
        tmp_path (Path): Pytest 提供的隔离模型目录。
        monkeypatch (pytest.MonkeyPatch): 用于替换真实磁盘身份检查。

    Raises:
        AssertionError: readiness 请求未绑定期望的服务模型。

    Returns:
        None: 此测试只检查服务期间的模型切换。
    """
    from play_sts2.inference import InferenceModelIdentityError
    from play_sts2.runtime import model_smoke

    merged_model, serving_model = _model_fixture(tmp_path)
    monkeypatch.setattr(
        model_smoke,
        "validate_serving_model",
        lambda *_args, **_kwargs: None,
    )

    def respond(request: httpx.Request) -> httpx.Response:
        """先确认目标模型，再模拟生成阶段切换回旧模型。

        Args:
            request (httpx.Request): readiness 流程发出的 HTTP 请求。

        Raises:
            AssertionError: 生成请求未绑定期望的服务模型。

        Returns:
            httpx.Response: 健康、目标模型列表或旧模型生成响应。
        """
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": str(serving_model.resolve())}]},
            )
        body = json.loads(request.content)
        assert body["model"] == str(serving_model.resolve())
        return httpx.Response(
            200,
            json={
                "model": str(tmp_path / "serving/demo-e2-mlx-8bit"),
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with pytest.raises(InferenceModelIdentityError):
        model_smoke.validate_model_service(
            "http://127.0.0.1:8900",
            artifact_id="demo-e3",
            merged_model=merged_model,
            serving_model=serving_model,
            enable_thinking=False,
            transport=httpx.MockTransport(respond),
        )


def test_installed_mlx_server_binds_requested_path_and_echoes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """锁定 MLX 0.31.x 的契约：显式路径触发该路径加载并原样回报。"""
    from types import SimpleNamespace

    server = pytest.importorskip("mlx_lm.server")
    old_model = tmp_path / "demo-e2-mlx-8bit"
    expected_model = tmp_path / "demo-e3-mlx-8bit"
    args = SimpleNamespace(
        model=str(old_model),
        adapter_path=None,
        draft_model=None,
        pipeline=False,
        trust_remote_code=False,
        chat_template=None,
    )
    provider = server.ModelProvider(args)
    loaded: list[tuple[str, object, object]] = []
    monkeypatch.setattr(
        provider,
        "_load",
        lambda model, adapter, draft: loaded.append((model, adapter, draft)),
    )

    provider.load(str(expected_model))

    assert loaded == [(str(expected_model), None, None)]
    handler = object.__new__(server.APIHandler)
    handler.requested_model = str(expected_model)
    handler.request_id = "chatcmpl-contract"
    handler.system_fingerprint = "test"
    handler.object_type = "chat.completion"
    handler.created = 0
    handler.stream = False
    response = handler.generate_response("ok", "stop", 1, 1)
    assert response["model"] == str(expected_model)


def _model_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """创建带轻量来源身份的合并与 MLX 模型目录。"""
    merged_model = tmp_path / "merged/demo-e3-merged"
    serving_model = tmp_path / "serving/demo-e3-mlx-8bit"
    merged_model.mkdir(parents=True)
    serving_model.mkdir(parents=True)
    merge_manifest = (
        json.dumps(
            {
                "schema_version": 1,
                "adapter": str(tmp_path / "adapters/demo-e3"),
                "output": str(merged_model.resolve()),
                "tokenizer_source": str(tmp_path / "adapters/demo-e3"),
                "source_sha256": {
                    "base/config.json": "a" * 64,
                    "adapter/train_manifest.json": "b" * 64,
                    "tokenizer/tokenizer.json": "c" * 64,
                },
            }
        )
        + "\n"
    ).encode()
    (merged_model / "merge_manifest.json").write_bytes(merge_manifest)
    (serving_model / "serving_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_id": "demo-e3",
                "source_model": str(merged_model.resolve()),
                "source_merge_manifest_sha256": hashlib.sha256(
                    merge_manifest
                ).hexdigest(),
                "quantization": {"bits": 8, "group_size": 64},
                "eos_token_id": 3,
                "thinking_template_sha256": {
                    "disabled": "a" * 64,
                    "enabled": "b" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    (serving_model / "config.json").write_text(
        json.dumps(
            {
                "eos_token_id": 3,
                "quantization": {"bits": 8, "group_size": 64},
            }
        ),
        encoding="utf-8",
    )
    return merged_model, serving_model
