"""W5D5 Python · 类与对象练习。

跑法：
    python -X utf8 classes_drill.py

目标：6/6 通过。
"""

from textstats import count_tokens, top_k


class TokenCounter:
    """把 token 列表和统计行为包成一个对象。"""

    def __init__(self, tokens):
        # TODO 1：保存 tokens 的副本，不要让外部列表后续修改影响对象。
        self.tokens=list(tokens)

    def counts(self):
        # TODO 2：返回 self.tokens 的词频字典。
        # 可以直接复用 count_tokens。
        return count_tokens(self.tokens)

    def top(self, k):
        # TODO 3：返回前 k 个 (word, count)。
        # 可以直接复用 top_k。
        return top_k(self.counts(),k)

    def __repr__(self):
        # TODO 4：返回类似 TokenCounter(tokens=3) 的调试字符串。
        return f"TokenCounter(tokens={len(self.tokens)})"
        raise NotImplementedError("TODO 4: __repr__")


def _check(name, got, want):
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}  want={want!r}")
    return ok


def _selftest():
    source = ["apple", "banana", "apple"]
    try:
        counter = TokenCounter(source)
    except Exception as e:
        print(f"[FAIL] TokenCounter 初始化: {type(e).__name__}: {e}")
        print("\n0/6 通过")
        return False

    results = []
    results.append(_check("counts", counter.counts(), {"apple": 2, "banana": 1}))
    results.append(_check("top", counter.top(1), [("apple", 2)]))

    source.append("cherry")
    results.append(_check(
        "copy input",
        counter.counts(),
        {"apple": 2, "banana": 1},
    ))

    results.append(_check(
        "repr",
        repr(counter),
        "TokenCounter(tokens=3)",
    ))

    empty_counter = TokenCounter([])
    results.append(_check("empty counts", empty_counter.counts(), {}))
    results.append(_check("empty top", empty_counter.top(5), []))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过")
    return passed == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)


# ================== 我的混淆总结 ==================
# （做完之后把你踩到的坑写在下面）
