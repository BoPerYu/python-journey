"""W5D6 Python · dataclass 实验记录练习。

跑法：
    python -X utf8 dataclass_drill.py

目标：6/6 通过。
"""

from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentResult:
    name: str
    accuracy: float
    macro_f1: float
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.accuracy>=0.8 and self.macro_f1>=0.7


def _check(name, got, want):
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}  want={want!r}")
    return ok


def _selftest():
    results = []

    a = ExperimentResult("tfidf-baseline", 0.86, 0.75)
    results.append(_check(
        "repr contains name",
        "tfidf-baseline" in repr(a),
        True,
    ))
    results.append(_check("passed", a.passed, True))

    b = ExperimentResult("weak-baseline", 0.70, 0.75)
    results.append(_check("not passed", b.passed, False))

    c = ExperimentResult("tfidf-baseline", 0.86, 0.75)
    results.append(_check("dataclass equality", a == c, True))

    e1 = ExperimentResult("one", 0.9, 0.8)
    e2 = ExperimentResult("two", 0.9, 0.8)
    e1.errors.append("short text")
    results.append(_check("independent default list", e2.errors, []))

    results.append(_check(
        "asdict",
        asdict(e1)["errors"],
        ["short text"],
    ))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过")
    return passed == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)


# ================== 我的混淆总结 ==================
# （做完之后把你踩到的坑写在下面）
