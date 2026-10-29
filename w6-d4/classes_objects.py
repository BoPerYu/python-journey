"""W6 class/object drill: instance state and a checked mutation."""


class StudyLog:
    category = "study"

    def __init__(self, title: str):
        self.title = title
        self.minutes = 0
        self.done = False

    def add_minutes(self, amount: int) -> None:
        if amount>0:
            self.minutes+=amount
        else:
            raise ValueError ("amount必须为正整数。")

    def finish(self) -> None:
        self.done=True

    def summary(self) -> str:
        if self.done:
            return f"{self.title} | {self.minutes}min | 完成"
        else:
            return f"{self.title} | {self.minutes}min | 进行中"


# ================== 自测（这一段不要改） ==================
def _check(name, got, want):
    ok = got == want and type(got) is type(want)
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    return ok


def _check_call(name, action, want):
    try:
        got = action()
    except Exception as error:
        print(f"[FAIL] {name}: {type(error).__name__}: {error}")
        return False
    return _check(name, got, want)


def _check_raises_value_error(name, action, log):
    before = (log.title, log.minutes, log.done)
    try:
        action()
    except ValueError:
        after = (log.title, log.minutes, log.done)
        ok = after == before
        print(f"[{'OK  ' if ok else 'FAIL'}] {name}: state_before={before!r} state_after={after!r}")
        return ok
    except Exception as error:
        print(f"[FAIL] {name}: unexpected {type(error).__name__}: {error}")
        return False

    print(f"[FAIL] {name}: ValueError was not raised")
    return False


def _selftest():
    results = []

    first = StudyLog("C++")
    second = StudyLog("Python")
    results.append(_check("class attribute is shared", StudyLog.category, "study"))
    results.append(_check("two instances have distinct titles", first.title != second.title, True))

    results.append(_check_call("first add returns None", lambda: first.add_minutes(25), None))
    results.append(_check_call("second add returns None", lambda: first.add_minutes(15), None))
    results.append(_check("add_minutes accumulates", first.minutes, 40))
    results.append(_check("other instance is unchanged", second.minutes, 0))

    results.append(_check_raises_value_error("zero is rejected", lambda: first.add_minutes(0), first))
    results.append(_check_raises_value_error("negative is rejected", lambda: first.add_minutes(-5), first))

    results.append(_check_call("unfinished summary", first.summary, "C++ | 40min | 进行中"))
    results.append(_check_call("finish returns None", first.finish, None))
    results.append(_check("finish marks only current object", first.done and not second.done, True))
    results.append(_check_call("finished summary", first.summary, "C++ | 40min | 完成"))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过" + ("" if passed == len(results) else "  ← 还没写对"))


if __name__ == "__main__":
    _selftest()


# ================== 我的混淆总结 ==================
# 实例属性、类属性、self、绑定方法、对象身份、非法输入后的状态一致性：
# （做完后自己写在这里）
