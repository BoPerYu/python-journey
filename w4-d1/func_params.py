# W4D6（2026-10-16）· 函数的参数进阶（6 个函数 · 8 个检查）
# --------------------------------------------------------------------------
# 讲义：打卡训练\W4\W4D6_Python_函数参数进阶.md
# 跑法：cd D:\Code\python-journey\w4-d1  ->  python -X utf8 func_params.py
# 目标：最后一行 "8/8 通过"
# --------------------------------------------------------------------------


def greet(name, greeting="你好"):
    """返回 "问候语，名字" 这样的字符串；不传 greeting 就用 "你好"。

    greet("小明")             -> "你好，小明"
    greet("小明", "早上好")    -> "早上好，小明"
    """
    # TODO：函数体你写
    a=f"{greeting}，{name}"
    return a


def total(*nums):
    """把所有传进来的数加起来；一个都没传就返回 0。

    total(1, 2, 3) -> 6
    total()        -> 0
    提示：nums 在函数里是【一个元组】，用 for 遍历它。
    """
    # TODO：函数体你写
    total=0
    for i in nums:
        total+=i
    return total


def describe(**info):
    """把关键字参数拼成 "键=值" 用逗号连接、按【键】排序的字符串。

    describe(b=2, a=1) -> "a=1, b=2"
    提示：info 在函数里是【一个字典】；昨天学的 sorted(info.items()) 正好用得上。
    """
    # TODO：函数体你写
    answer=dict(sorted(info.items(),key=lambda k:k[0]))
    a=[]
    for i,x in answer.items():
        a.append(f"{i}={x}")
    return ", ".join(a) 


def apply_twice(func, x):
    """返回 func(func(x))。（练"函数可以当参数传"）

    apply_twice(lambda x: x * 2, 3) -> 12
    """
    # TODO：函数体你写
    a=func(func(x))
    return a


def call_with_list(func, args_list):
    """把 args_list 里的东西拆成一个个参数，喂给 func。

    call_with_list(lambda a, b: a + b, [2, 3]) -> 5
    提示：调用时用 * 解包。
    """
    # TODO：函数体你写
    return func(*args_list)


def make_multiplier(n):
    """返回一个【函数】：那个函数接受一个数，返回它乘 n 的结果。

    make_multiplier(3)(5) -> 15
    提示：返回 lambda。
    """
    # TODO：函数体你写
    return lambda x:x*n


# ================== 自检（这一段不要改） ==================
def _check(name, got, want):
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}  want={want!r}")
    return ok


def _selftest():
    cases = [
        ("1a greet 不传问候语",  lambda: greet("小明"),                 "你好，小明"),
        ("1b greet 传了问候语",  lambda: greet("小明", "早上好"),        "早上好，小明"),
        ("2a total(1,2,3)",      lambda: total(1, 2, 3),                6),
        ("2b total()",           lambda: total(),                       0),
        ("3  describe(b=2,a=1)", lambda: describe(b=2, a=1),            "a=1, b=2"),
        ("4  apply_twice",       lambda: apply_twice(lambda x: x * 2, 3), 12),
        ("5  call_with_list",    lambda: call_with_list(lambda a, b: a + b, [2, 3]), 5),
        ("6  make_multiplier",   lambda: make_multiplier(3)(5),          15),
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
