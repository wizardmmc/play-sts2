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

## 其他
机器、服务器等本机私有信息写在 `AGENTS.local.md`，该文件已被 Git 忽略。
不要罗列实际很难遇到的bug
不要过度工程化和防御性编程，例如滥用版本哈希
做好checkpoint，同时step不要太小，避免把磁盘写坏
