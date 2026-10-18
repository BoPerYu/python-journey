# W4D2（2026-10-13）· Python 第 8 课：作用域与传参语义（6 个小任务）
# --------------------------------------------------------------------------
# 讲义：打卡训练\W4\W4D2_Python第8课_作用域与传参语义.md
# 跑法：
#     cd D:\Code\python-journey\w4-d1
#     python -X utf8 scope_params.py
# 全部写对时最后一行是 "6/6 通过"。
#
# 规矩照旧：下面 6 个函数的**函数体你自己写**（现在都是 `...` 占位）。
# --------------------------------------------------------------------------


counter = 0          # 模块级变量，任务 1、2 要用它（别删）


def counter_bad():
    """任务 1：就写一行  counter = counter + 1  （故意不写 global）

    写好之后这个函数会抛 UnboundLocalError —— 那是【预期结果】，
    自检器就是靠它判断你有没有真的踩到。讲义 2.2 讲了为什么会这样。
    """
    counter = counter+1


def counter_good():
    """任务 2：用 global 声明，然后把外面的 counter 加 1"""
    global counter
    counter+=1


def append_one(v):
    """任务 3：在函数里写  v.append(1)   （改"对象"）
    问：调用方那个列表会变吗？
    会。我能不能这么理解，（）里的作为形参，v进来也会指向同样的地址
    """
    v.append(1)


def plus_one(v):
    """任务 4：在函数里写  v = v + [1]   （给"名字"重新赋值）

    问：和任务 3 有什么区别？外面会变吗？
    任务三是直接在原来的基础上改对象，这个是直接新改了list的内容。外面应该不会变，因为v=给v贴了一个新的地址，指向的是v+[i]的那一个数组
    """
    v=v+[1]



def iadd_one(v):
    """任务 5：在函数里写  v += [1]

    问：和任务 4 只差一个加号，结果一样吗？（讲义 3.3 那条反直觉的）
    按照你的说法是不会变的，但是我不太清楚这个和任务4的函数在执行逻辑上的具体区别，请展开讲讲。因为如果v是int的话，那v+=1和v=v+1都不行不是吗
    """
    v+=[1]


def bump(n):
    """任务 6：在函数里写  n = n + 1     （n 是 int，不可变）

    问：调用方那个整数会变吗？
    不会，但承接我上边的问题，我不理解他和list运算的区别。
    """
    n=n+1


# ================== 自检（这一段不要改） ==================
# 每个任务都套在 try 里：某个函数写崩了也只是这一行 FAIL，不会把整个自检带崩。
def _check(name, got, want):
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}  want={want!r}")
    return ok


def _selftest():
    global counter
    r = []

    counter = 0
    try:
        counter_bad()
        got = "没有报错"
    except UnboundLocalError:
        got = "UnboundLocalError"
    except Exception as e:
        got = type(e).__name__
    r.append(_check("1  counter_bad() 的结果", got, "UnboundLocalError"))

    counter = 0
    try:
        counter_good()
        counter_good()
        got = counter
    except Exception as e:
        got = f"抛异常 {type(e).__name__}"
    r.append(_check("2  调两次 counter_good() 之后的 counter", got, 2))

    a = [1]
    try:
        append_one(a)
        got = a
    except Exception as e:
        got = f"抛异常 {type(e).__name__}"
    r.append(_check("3  append_one 之后的 a（改对象 → 看得见）", got, [1, 1]))

    b = [1]
    try:
        plus_one(b)
        got = b
    except Exception as e:
        got = f"抛异常 {type(e).__name__}"
    r.append(_check("4  plus_one 之后的 b（重新绑定 → 看不见）", got, [1]))

    c = [1]
    try:
        iadd_one(c)
        got = c
    except Exception as e:
        got = f"抛异常 {type(e).__name__}"
    r.append(_check("5  iadd_one 之后的 c（list 的 += 原地改 → 看得见）", got, [1, 1]))

    y = 5
    try:
        bump(y)
        got = y
    except Exception as e:
        got = f"抛异常 {type(e).__name__}"
    r.append(_check("6  bump(y) 之后的 y（int 不可变）", got, 5))

    passed = sum(r)
    print(f"\n{passed}/{len(r)} 通过" + ("" if passed == len(r) else "  ← 还没写完，回去看 FAIL 那几行"))
    return passed == len(r)


if __name__ == "__main__":
    _selftest()
