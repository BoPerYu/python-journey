# W4D1（2026-10-12）· 用 Python 的 list 写一个栈
# ------------------------------------------------------------------
# 约定：栈 = 一个 list。栈底在下标 0，**栈顶是最后一个元素**。
#      只用已经学过的语法：list / def / if / else / len()。
#      （类还没学 —— 那是 W6 的内容，所以这里写成 5 个函数，不写 class。）
#
# 规则：下面 5 个函数的**函数体你自己写**（现在写的是 `...` 占位）。
#      写完之后运行一次：  python -X utf8 stack_list.py
#      （`-X utf8` 是为了让中文不因控制台代码页而乱码；不带也能跑。）
#      必须看到最后一行是 "13/13 通过" 才算写完。
#      —— 这一步是在练你 W3 暴露的头号问题："写完不验证"（5 次静默错误，编译器一次都救不了你）。
#
# 三个对照题（写在文件最下方「我的对照笔记」里，两三句就够）：
#   Q1 C++ 里 top() 写成 `int& top()`，就能 `s.top() = 999;` 直接改栈里那个格子。
#      Python 这边 `top(s) = 999` 是什么下场？为什么？
#   Q2 Python 里 `def push(stack, value)` 里的 `stack`，传进来的是**副本**还是**同一块存储**？
#      证据：`a = [1]; push(a, 2); print(a)`
#   Q3 W3 你实测过"C++ 基类装不下派生类的东西"（对象切片）。
#      Python 里 `x = [1, 2]; y = x` 会不会发生类似的事？
# ------------------------------------------------------------------

def push(stack, value):
    """把 value 压到栈顶（就地改 stack）。没有返回值。"""
    stack.append(value)


def pop(stack):
    """弹出并返回栈顶元素；栈空时返回 None。
    
    注意：**不要**让它抛 IndexError（那是 list.pop() 的默认行为），
    所以你必须先判断空不空。
    """
    if (len(stack)==0):
        return None
    else:
        return stack.pop()


def top(stack):
    """看一眼栈顶，但**不弹**；栈空时返回 None。"""
    if(len(stack)==0):
        return None
    else:
        return stack[len(stack)-1]


def is_empty(stack):
    """栈空返回 True，否则返回 False。"""
    if(len(stack)==0):
        return True
    else:
        return False


def size(stack):
    """返回元素个数。"""
    return len(stack)


# ================== 自检（这一段不要改） ==================
def _check(name, got, want):
    # 注意 type 一起比：Python 里 True == 1，光比 == 会把 True 当成 1 放过去
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def _selftest():
    r = []
    s = []
    r.append(_check("空栈 is_empty", is_empty(s), True))
    r.append(_check("空栈 size", size(s), 0))
    r.append(_check("空栈 pop -> None", pop(s), None))
    r.append(_check("空栈 top -> None", top(s), None))

    push(s, 10)
    push(s, 20)
    push(s, 30)
    r.append(_check("压 3 个后 size", size(s), 3))
    r.append(_check("top 看到 30", top(s), 30))
    r.append(_check("top 不弹栈", size(s), 3))
    r.append(_check("pop -> 30", pop(s), 30))
    r.append(_check("pop 后 size", size(s), 2))
    r.append(_check("再 pop -> 20", pop(s), 20))
    r.append(_check("再 pop -> 10", pop(s), 10))
    r.append(_check("弹空后 is_empty", is_empty(s), True))
    push(s, "a")
    push(s, "b")
    r.append(_check("不同类型也能压", pop(s), "b"))

    passed = sum(r)
    print(f"\n{passed}/{len(r)} 通过" + ("" if passed == len(r) else "  ← 还没写完，回去看 FAIL 那几行"))
    return passed == len(r)


if __name__ == "__main__":
    _selftest()


# ================== 我的对照笔记 ==================
# Q1 不知道，请详细讲讲
# Q2 直接是存储
# Q3 不会，因为y=x等于是把y和x共同指向同一个栈了
