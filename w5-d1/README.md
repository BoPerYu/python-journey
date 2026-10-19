# w5-d1 · Python 模块与包

今天的练习：把“读文件 → 计数 → 排序 → 输出”拆成一个包。

## 目录

```text
w5-d1\
├── textstats\
│   ├── __init__.py      # 包的入口，导出公开函数
│   ├── io.py            # 文件 -> token 列表
│   ├── stats.py         # token -> 词频 -> Top-K
│   └── __main__.py      # python -m textstats 的入口
├── data\sample.txt
├── selftest.py
└── README.md
```

## 跑法

在 `D:\Code\python-journey\w5-d1` 目录下：

```powershell
python -X utf8 selftest.py
python -m textstats
```

当前骨架会抛 `NotImplementedError`；填完三个 TODO 后，自检目标是 `7/7 通过`。

## 今天要观察的三件事

1. `python selftest.py` 和 `python -m textstats` 的当前目录要求有什么不同？
2. `textstats/__init__.py` 为什么能让 `from textstats import count_tokens` 成立？
3. `textstats/__main__.py` 里为什么用 `from . import ...`，而不是 `from textstats import ...`？
