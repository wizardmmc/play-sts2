"""解析和渲染单实体 Markdown 游戏知识条目。"""

from dataclasses import dataclass


class KnowledgeFormatError(ValueError):
    """表示 Markdown 知识条目不符合最小 frontmatter 契约。"""


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    """表示一个带来源信息的可读游戏知识条目。

    Args:
        metadata (dict[str, str]): 保持原顺序的扁平 frontmatter 字段。
        body (str): 不包含 frontmatter 的 Markdown 正文。
    """

    metadata: dict[str, str]
    body: str

    @property
    def object_id(self) -> str:
        """返回知识实体的稳定游戏 ID。

        Returns:
            str: ``id`` frontmatter 字段。
        """
        return self.metadata["id"]

    @property
    def name(self) -> str:
        """返回知识实体的游戏显示名称。

        Returns:
            str: ``name`` frontmatter 字段。
        """
        return self.metadata["name"]

    @property
    def source(self) -> str:
        """返回实体事实的来源类别。

        Returns:
            str: 例如 ``web_wiki`` 或 ``mod_export``。
        """
        return self.metadata["source"]

    def to_markdown(self) -> str:
        """渲染为可直接写盘的单实体 Markdown。

        Raises:
            KnowledgeFormatError: frontmatter 值包含换行，无法安全平铺。

        Returns:
            str: 以换行结尾的完整 Markdown 文本。
        """
        lines = ["---"]
        for key, value in self.metadata.items():
            if "\n" in value or "\r" in value:
                raise KnowledgeFormatError(f"frontmatter 字段不能包含换行: {key}")
            lines.append(f"{key}: {value}")
        lines.extend(("---", self.body.strip()))
        return "\n".join(lines).rstrip() + "\n"


def parse_knowledge_entry(text: str) -> KnowledgeEntry:
    """解析成扁平 frontmatter 与 Markdown 正文。

    Args:
        text (str): 待解析的完整 Markdown 文本。

    Raises:
        KnowledgeFormatError: 缺少边界、字段语法错误或必要字段为空。

    Returns:
        KnowledgeEntry: 保留 frontmatter 原始字符串值的知识条目。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise KnowledgeFormatError("知识条目缺少 frontmatter 起始边界")
    try:
        end = next(
            index for index, line in enumerate(lines[1:], start=1) if line == "---"
        )
    except StopIteration as exc:
        raise KnowledgeFormatError("知识条目缺少 frontmatter 结束边界") from exc

    metadata: dict[str, str] = {}
    for line in lines[1:end]:
        key, separator, value = line.partition(":")
        if not separator or not key.strip():
            raise KnowledgeFormatError(f"无效 frontmatter 字段: {line}")
        metadata[key.strip()] = value.strip()
    for field in ("id", "name", "type", "source"):
        if not metadata.get(field):
            raise KnowledgeFormatError(f"知识条目缺少字段: {field}")
    return KnowledgeEntry(metadata=metadata, body="\n".join(lines[end + 1 :]).strip())
