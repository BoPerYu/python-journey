# W4D1（2026-10-12）· Python 第 7 课：列表推导式（10 个函数级小练习）
# --------------------------------------------------------------------------
# 讲义：打卡训练\W4\W4D1_Python第7课_列表推导式.md
# 跑法：
#     cd D:\Code\python-journey\w4-d1
#     python -X utf8 comprehension.py
# 全部写对时会看到最后一行 "12/12 通过"。
#
# 规矩还是那两条：
#   1) 下面每个函数的**函数体你自己写**（现在都是 `...` 占位）。
#   2) 每写一个就跑一次 —— 自检器会告诉你哪个没对。
#      这是冲着你 W3 那条"写完不验证"来的（5 次静默错误，编译器一次都没救你）。
# --------------------------------------------------------------------------
nums = [1,2,3,4,5]

def squares(n):
    """返回 [1², 2², ..., n²]。n 至少是 1。

    squares(5) -> [1, 4, 9, 16, 25]
    提示：range(1, n + 1) —— 注意 n+1，差一格是 Python 最常见的错。
    """
    return[x**2 for x in range(1,n+1)]



def evens(nums):
    """只保留偶数，顺序不变。

    evens([1, 2, 3, 4, 5, 6]) -> [2, 4, 6]
    """
    return [x for x in nums if x % 2 == 0]


def strip_all(words):
    """把每个词首尾的空格去掉。

    strip_all(["  a ", "b  "]) -> ["a", "b"]
    """
    return [x.strip() for x in words]


def long_words(words, min_len):
    """只保留长度 >= min_len 的词（原样，不去空格）。

    long_words(["hi", "hello", "world", "a"], 4) -> ["hello", "world"]
    """
    return [x for x in words if len(x)>=min_len]


def upper_all(words):
    """全部转成大写。

    upper_all(["ab", "Cd"]) -> ["AB", "CD"]
    """
    return [x.upper() for x in words]


def pass_fail(scores):
    """分数 >= 60 记成 "及格"，否则 "不及格"。

    pass_fail([59, 60, 100]) -> ["不及格", "及格", "及格"]
    提示：这一题要用**三元表达式**，而且它写在 for 的【前面】：
          ["及格" if s >= 60 else "不及格" for s in scores]
          —— 和上一题的"筛选"（if 写在后面）位置不同，这是最容易混的地方。
    """
    return["及格" if x >=60 else "不及格" for x in [59,60,100]]


def flatten(rows):
    """把二维的拍成一维。

    flatten([[1, 2], [3], [4, 5]]) -> [1, 2, 3, 4, 5]
    提示：两个 for，写在同一层，顺序 = 你把嵌套循环摊开的顺序。
    """
    return [x for row in rows for x in row]


def pairs(a, b):
    """返回所有 (x, y) 组合（笛卡尔积），x 来自 a、y 来自 b。

    pairs([1, 2], ["a", "b"]) -> [(1, "a"), (1, "b"), (2, "a"), (2, "b")]
    """
    return [(x,y) for x in a for y in b]


def word_lengths(words):
    """返回 {词: 长度} 的字典。

    word_lengths(["a", "bb"]) -> {"a": 1, "bb": 2}
    提示：这一题是**字典推导式**，用花括号 + 冒号：{键: 值 for ...}
    """
    return {a:len(a)  for a in words}


def fizz_short(n):
    """综合题：返回 1 到 n 的列表（元素都是字符串），规则：
        能同时被 3 和 5 整除 -> "FizzBuzz"
        只能被 3 整除        -> "Fizz"
        只能被 5 整除        -> "Buzz"
        其它                 -> 这个数本身的字符串，比如 "7"

    fizz_short(5)  -> ["1", "2", "Fizz", "4", "Buzz"]
    fizz_short(15) -> [..., "14", "FizzBuzz"]（长度 15，最后一个是 "FizzBuzz"）
    提示：三元表达式可以连着写，但**从前往后**判断：
          先判 15，再判 3，再判 5，最后兜底 —— 顺序错了结果就错。
          用 str(x) 把数字变成字符串。
    """
    return ["FizzBuzz" if x%3==0 and x%5==0 else "Fizz" if x%3==0 and x%5!=0 else "Buzz" if x%3!=0 and x%5==0 else str(x) for x in range(1,n+1)]


# ================== 自检（这一段不要改） ==================
# 每个用例都是"一个函数调用 + 期望值"。调用会放进 try 里：
# 就算你某个函数还没写、或者写崩了，也只是这一行 FAIL，
# 不会把整个自检器带崩（不然你就只能看到一片 Traceback，看不到别的对错）。
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
        ("1  squares(5)",        lambda: squares(5),                        [1, 4, 9, 16, 25]),
        ("2  evens([1..6])",     lambda: evens([1, 2, 3, 4, 5, 6]),         [2, 4, 6]),
        ("2b evens([])",         lambda: evens([]),                         []),
        ("3  strip_all",         lambda: strip_all(["  a ", "b  "]),        ["a", "b"]),
        ("4  long_words",        lambda: long_words(["hi", "hello", "world", "a"], 4), ["hello", "world"]),
        ("5  upper_all",         lambda: upper_all(["ab", "Cd"]),           ["AB", "CD"]),
        ("6  pass_fail",         lambda: pass_fail([59, 60, 100]),          ["不及格", "及格", "及格"]),
        ("7  flatten",           lambda: flatten([[1, 2], [3], [4, 5]]),    [1, 2, 3, 4, 5]),
        ("7b flatten([])",       lambda: flatten([]),                       []),
        ("8  pairs",             lambda: pairs([1, 2], ["a", "b"]),
                                 [(1, "a"), (1, "b"), (2, "a"), (2, "b")]),
        ("9  word_lengths",      lambda: word_lengths(["a", "bb"]),         {"a": 1, "bb": 2}),
        ("10 fizz_short(5)",     lambda: fizz_short(5),                     ["1", "2", "Fizz", "4", "Buzz"]),
        ("10b fizz_short(15)",   lambda: fizz_short(15),
                                 ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz",
                                  "11", "Fizz", "13", "14", "FizzBuzz"]),
    ]
    r = [_check(n, f, w) for n, f, w in cases]
    passed = sum(r)
    print(f"\n{passed}/{len(r)} 通过" + ("" if passed == len(r) else "  ← 还没写完，回去看 FAIL 那几行"))
    return passed == len(r)


if __name__ == "__main__":
    _selftest()
