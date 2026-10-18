# W4D3（2026-10-14）· 手写 split vs 内置 split
# --------------------------------------------------------------------------
# 讲义：打卡训练\W4\W4D3_Python_手写split与内置split.md
# 跑法：cd D:\Code\python-journey\w4-d1  ->  python -X utf8 split_compare.py
# 目标：最后一行 "6/6 通过"
# --------------------------------------------------------------------------


def split_manual(s, delim):
    """手写切词：按 delim 切，【跳过空片段】。不许用 s.split()。

    模仿今天 C++ 那套流程（讲义第一节的对照表）：
        start = 0
        pos = s.find(delim, start)        # 从 start 往后找
        while pos != -1:                  # Python 里"没找到"是 -1
            片段 = s[start:pos]
            if 片段 != "":  收进结果       # 空的就跳过
            start = pos + 1               # 跳过分隔符本身
            pos = s.find(delim, start)    # ⚠️ 一定要重新找，不然死循环
        最后一段 = s[start:]
        if 最后一段 != "": 收进结果
        return 结果

    例：
        split_manual("the quick brown fox", " ") -> ["the","quick","brown","fox"]
        split_manual("a,,b", ",")                -> ["a","b"]
        split_manual("  hi  ", " ")              -> ["hi"]
        split_manual("", " ")                    -> []
    """
    # TODO：函数体你写
    answer=[]
    start=0
    pos=s.find(delim,start)
    while pos!=-1:
        if pos-start!=0:
            answer.append(s[start:pos])
        start=pos+1
        pos=s.find(delim,start)
    p=(s[start:])
    if p!="":
        answer.append(p)
    return answer


# ================== 自检（这一段不要改） ==================
def _check(name, got, want):
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}  want={want!r}")
    return ok


def _selftest():
    cases = [
        ("1  一句话按空格",      ("the quick brown fox", " "), ["the", "quick", "brown", "fox"]),
        ("2  连续分隔符跳空",    ("a,,b", ","),               ["a", "b"]),
        ("3  两头空白跳空",      ("  hi  ", " "),             ["hi"]),
        ("4  空串",              ("", " "),                   []),
        ("5  连续三个分隔符",    ("x,,,y", ","),              ["x", "y"]),
        ("6  分隔符在结尾",      ("a,b,", ","),               ["a", "b"]),
    ]
    r = []
    for name, (text, d), want in cases:
        try:
            r.append(_check(name, split_manual(text, d), want))
        except Exception as e:
            print(f"[FAIL] {name} 抛异常 {type(e).__name__}: {e}")
            r.append(False)

    print("\n======== 观察：你的手写版 vs 内置 split（两列并排，别搞混）========")
    print("   【左列】是你写的 split_manual 的结果；【右列】是 Python 内置 text.split(delim) 的结果。")
    print("   右列里那些空串 '' 是【内置版】的行为（它保留空片段），不是你代码的问题。\n")
    print(f"   {'输入':<24}{'你的 split_manual':<30}{'内置 text.split(delim)'}")
    for text, d in [("the quick brown fox", " "), ("a,,b", ","), ("  hi  ", " "), ("", " ")]:
        print(f"   {text!r:<24}{str(split_manual(text, d)):<30}{text.split(d)}")
    print()
    print(f"   {'  hi  '!r:<24}{'（这是另一件事）':<30}{'  hi  '.split()}   ← 不带参数：按任意空白切，空片段丢掉")
    print(f"   {''!r:<24}{str(split_manual('', ' ')):<30}{''.split()}   ← 这样才是 []")

    passed = sum(r)
    print(f"\n{passed}/{len(r)} 通过" + ("" if passed == len(r) else "  ← 还没写完/写错了"))
    return passed == len(r)


if __name__ == "__main__":
    _selftest()


# ================== 我的对照结论（这三问你自己答） ==================
# 问 1：手写版对 "a,,b" 给 ['a','b']，内置 split(',') 给 ['a','','b']。
#       两个都对吗？什么时候你要用哪一个？
# 答：
#
# 问 2：为什么 "".split(",") 是 [''] 而不是 []？说说它是怎么切出来的。
# 答：
#
# 问 3：今天 C++ 那个"返回 vector 会不会拷贝"的问题，在 Python 里为什么不存在？
# 答：
