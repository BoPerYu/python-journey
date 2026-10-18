# W4D5（2026-10-15）· 元组与循环技巧（7 个函数 · 11 个检查）
# --------------------------------------------------------------------------
# 讲义：打卡训练\W4\W4D5_Python_循环技巧.md
# 跑法：cd D:\Code\python-journey\w4-d1  ->  python -X utf8 loop_tools.py
# 目标：最后一行 "11/11 通过"
# --------------------------------------------------------------------------


def index_of_all(nums, target):
    """返回 target 在 nums 里出现过的【所有下标】（用 enumerate）。

    index_of_all([1, 2, 1, 3, 1], 1) -> [0, 2, 4]
    index_of_all([1, 2], 9)          -> []
    """
    # TODO：函数体你写
    a=[]
    for i,x in enumerate(nums):
        if x==target:
            a.append(i)
    return a


def pair_up(a, b):
    """用 zip 配成 [(a0,b0), (a1,b1), ...]；长度不等就截断到最短。

    pair_up([1, 2, 3], ["a", "b"]) -> [(1, "a"), (2, "b")]
    """
    # TODO：函数体你写
    a=list(zip(a,b))
    return a


def sorted_items_by_key(d):
    """返回 (键, 值) 列表，按【键】从小到大。（d.items() + sorted）

    sorted_items_by_key({"b": 2, "a": 1}) -> [("a", 1), ("b", 2)]
    """
    # TODO：函数体你写
    return sorted(d.items(), key=lambda kv: kv[0])


def top_n_by_value(d, n):
    """返回 (键, 值) 列表，按【值】从大到小，只取前 n 个。

    top_n_by_value({"a": 3, "b": 5, "c": 1}, 2) -> [("b", 5), ("a", 3)]
    提示：sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]
    """
    # TODO：函数体你写
    a=sorted(d.items(), key=lambda kv: kv[1],reverse=True)
    b=[]
    for i in range(n):
        b.append(a[i])
    return b


def sort_by_length(words):
    """按字符串【长度】从小到大排（长度一样时保持原先后）。

    sort_by_length(["ccc", "a", "bb"]) -> ["a", "bb", "ccc"]
    提示：sorted 的 key 参数。
    """
    # TODO：函数体你写
    a=sorted(words,key=len)
    return a


def unequal_positions(a, b):
    """返回两个序列【不相等】的位置下标（enumerate + zip 合体）。

    unequal_positions([1, 2, 3], [1, 5, 3]) -> [1]
    unequal_positions([1, 2], [1, 2])       -> []
    """
    # TODO：函数体你写
    z=list(zip(a,b))
    answer=[]
    for i,(x,y) in enumerate(z):
        if x!=y:
            answer.append(i)
    return answer


def min_max(values):
    """返回一个【元组】(最小值, 最大值)。

    min_max([3, 1, 2]) -> (1, 3)
    min_max([5])       -> (5, 5)
    """
    # TODO：函数体你写
    a=sorted(values)
    return (a[0], a[-1])


# ================== 自检（这一段不要改） ==================
def _check(name, got, want):
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}  want={want!r}")
    return ok


def _selftest():
    cases = [
        ("1a index_of_all 有多个", lambda: index_of_all([1, 2, 1, 3, 1], 1), [0, 2, 4]),
        ("1b index_of_all 找不到", lambda: index_of_all([1, 2], 9), []),
        ("2  pair_up 截断到最短",  lambda: pair_up([1, 2, 3], ["a", "b"]), [(1, "a"), (2, "b")]),
        ("3  sorted_items_by_key", lambda: sorted_items_by_key({"b": 2, "a": 1}), [("a", 1), ("b", 2)]),
        ("4  top_n_by_value",      lambda: top_n_by_value({"a": 3, "b": 5, "c": 1}, 2), [("b", 5), ("a", 3)]),
        ("5a sort_by_length",      lambda: sort_by_length(["ccc", "a", "bb"]), ["a", "bb", "ccc"]),
        ("5b sort_by_length 等长", lambda: sort_by_length(["bb", "cc", "a"]), ["a", "bb", "cc"]),
        ("6a unequal_positions 有", lambda: unequal_positions([1, 2, 3], [1, 5, 3]), [1]),
        ("6b unequal_positions 无", lambda: unequal_positions([1, 2], [1, 2]), []),
        ("7a min_max 多个",        lambda: min_max([3, 1, 2]), (1, 3)),
        ("7b min_max 一个",        lambda: min_max([5]), (5, 5)),
    ]
    r = []
    for name, fn, want in cases:
        try:
            r.append(_check(name, fn(), want))
        except Exception as e:
            print(f"[FAIL] {name} 抛异常 {type(e).__name__}: {e}")
            r.append(False)
    passed = sum(r)
    print(f"\n{passed}/{len(r)} 通过" + ("" if passed == len(r) else "  ← 还没写完/写错了"))
    return passed == len(r)


if __name__ == "__main__":
    _selftest()


# ================== 我的混淆总结 ==================
# （做完之后把你踩到的坑写在下面）
