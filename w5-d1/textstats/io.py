"""负责“把文件变成 token 列表”。"""


def read_tokens(path):
    """读取 UTF-8 文本文件，按空白切分成 token 列表。

    例：文件内容 "the fox\\nthe dog\\n" -> ["the", "fox", "the", "dog"]

    提示：
    - 用 with open(path, "r", encoding="utf-8") as f
    - 用 f.read() 一次读进来，再 .split()
    - 本版本不处理标点、大小写和中文分词
    """
    # TODO 1：实现这个函数，然后删掉下面的 raise
    with open(path,"r",encoding="utf-8") as f:
        a=f.read()
        return a.split()