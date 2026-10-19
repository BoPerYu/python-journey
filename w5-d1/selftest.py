"""W5D1 模块/包练习的自检。

跑法（在 D:\\Code\\python-journey\\w5-d1 目录下）：
    python -X utf8 selftest.py
目标：
    7/7 通过
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from textstats import count_tokens, read_tokens, top_k


def _check(name, fn, want):
    try:
        got = fn()
    except Exception as e:
        print(f"[FAIL] {name}: 抛异常 {type(e).__name__}: {e}")
        return False
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}  want={want!r}")
    return ok


def _selftest():
    cases = [
        (
            "count_tokens 基本计数",
            lambda: count_tokens(["apple", "banana", "apple"]),
            {"apple": 2, "banana": 1},
        ),
        (
            "count_tokens 空列表",
            lambda: count_tokens([]),
            {},
        ),
        (
            "top_k 频次降序",
            lambda: top_k({"apple": 3, "banana": 2, "cherry": 1}, 2),
            [("apple", 3), ("banana", 2)],
        ),
        (
            "top_k 同频按单词升序",
            lambda: top_k({"b": 2, "a": 2, "c": 1}, 3),
            [("a", 2), ("b", 2), ("c", 1)],
        ),
        (
            "top_k 频次优先于字典序",
            lambda: top_k({"z": 3, "a": 2, "m": 1}, 3),
            [("z", 3), ("a", 2), ("m", 1)],
        ),
        (
            "top_k k 大于词数",
            lambda: top_k({"x": 1}, 5),
            [("x", 1)],
        ),
    ]

    results = []
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "tokens.txt"
        p.write_text("the fox\nthe dog\n", encoding="utf-8")
        cases.append((
            "read_tokens 读文件并切 token",
            lambda p=p: read_tokens(p),
            ["the", "fox", "the", "dog"],
        ))
        results = [_check(name, fn, want) for name, fn, want in cases]

    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过")
    return passed == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
