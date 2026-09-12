# 仓库开发约定

## 注释与文档字符串

- 所有 Python 模块、类、函数和方法都必须有中文 docstring；包括私有辅助函数、
  嵌套函数、异常类构造方法和协议方法。
- Python docstring 使用 Ruff 默认格式化能够稳定保持的 Google 风格：按需使用
  `Args:`、`Raises:`、`Returns:` 或 `Yields:`，参数类型写在参数名后的括号中。
  完成修改后运行 `uv run ruff format .` 和 `uv run ruff check .`。
- C# 类、record 和方法使用中文 XML 文档注释；参数、返回值及重要异常应写明
  契约。行内注释也使用中文，并只解释非显然的原因、边界或兼容处理。
- 注释不能只是复述赋值、循环或函数名；行为变化必须由测试表达，不能只靠注释
  声明。

## 数据产物边界

- `src/play_sts2/game_knowledge/curated/v0.107.1.json` 是进入离线重建的版本化
  curated 规则。
- 当前知识链路为：固定版本 Mod 原始导出与受控补充 → rebuild 时应用 curated
  → canonical `mod_export` → `generated-v0.107.1` 多问法候选 → 单独确定 E3
  混合及 train/dev/test 划分后再写入 `data/datasets/sft/`。
- 不要直接修补生成的 JSONL 或 SFT 分卷；应修正 Mod 导出、curated 规则、受控
  补录数据或生成代码后重新构建。

## 可用资源
实验室的5卡A100服务器
Host server
  HostName REDACTED
    User REDACTED
  Port 22
注：本质上这是我们的个人兴趣项目，所以最好单卡跑，如果没有别人用，允许多卡跑
vllm建议在gpu 0上跑，训练可以在gpu1-4上跑，因为卡1-2，3-4之间有NVLINK，方便多卡训练，效率高一点
卡0是没有的，只能跑单卡训练，所以没啥影响

## 其他
不要罗列实际很难遇到的bug
不要过度工程化和防御性编程，例如滥用版本哈希
本项目纯粹个人娱乐，不算前沿模型研究，请勿降智
