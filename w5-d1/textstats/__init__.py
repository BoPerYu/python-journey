"""textstats：W5D1 的模块/包练习。

这个文件让 textstats 成为一个包，并把常用函数从子模块里导出。
"""

from .io import read_tokens
from .stats import count_tokens, top_k

__all__ = ["read_tokens", "count_tokens", "top_k"]
