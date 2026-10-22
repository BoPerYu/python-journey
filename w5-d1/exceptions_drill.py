"""W5D2 异常处理练习。
跑法（在 D:\\Code\\python-journey\\w5-d1 目录下）：
    python -X utf8 exceptions_drill.py

目标：9/9 通过。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from textstats import read_tokens


def safe_int(text):
    """把 text 转成 int；转换失败返回 None。
    例：
      "42"  -> 42
      "abc" -> None
      "3.5" -> None

    提示：只捕获 ValueError，不要用裸 except。
    """
    # TODO 1：try / except ValueError / return
    try:
        value=int(text)
    except ValueError as e:
        return None
    return value

def parse_positive_int(text):
    """把 text 转成正整数；非法输入或 <= 0 时抛 ValueError。

    例：
      "7"   -> 7
      "0"   -> ValueError
      "-1"  -> ValueError
      "abc" -> ValueError

    提示：
    - int("abc") 自己会抛 ValueError；
    - 转成整数后再检查 <= 0，并主动 raise ValueError("...")。
    """
    # TODO 2：实现转换、检查和 raise
    value=int(text)
    if value<=0:
        raise ValueError("必须为正整数")
    return value



def describe_file_error(path):
    """读取 token 文件，返回 "ok" / "missing" / "encoding" / "os"。

    要求：
    - FileNotFoundError -> "missing"
    - UnicodeDecodeError -> "encoding"
    - 其他 OSError -> "os"
    - 成功 -> "ok"

    提示：except 的顺序要从具体到宽泛；
          FileNotFoundError 是 OSError 的子类。
    """
    # TODO 3：try read_tokens(path) + 三个 except + else
    try:
        with open(path,"r",encoding="utf-8") as f:
            text=f.read()
    except FileNotFoundError:
        return "missing"
    except UnicodeDecodeError:
        return "encoding"
    except OSError:
        return "os"
    return "ok"


def _check(name, fn, want):
    try:
        got = fn()
    except Exception as e:
        print(f"[FAIL] {name}: 抛异常 {type(e).__name__}: {e}")
        return False
    ok = (got == want) and (type(got) is type(want))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: got={got!r}  want={want!r}")
    return ok


def _check_raises(name, fn, want_type):
    try:
        fn()
    except Exception as e:
        ok = isinstance(e, want_type)
        print(f"[{'OK  ' if ok else 'FAIL'}] {name}: 抛出 {type(e).__name__}  want={want_type.__name__}")
        return ok
    print(f"[FAIL] {name}: 没有抛异常，want={want_type.__name__}")
    return False


def _selftest():
    results = [
        _check("safe_int 42", lambda: safe_int("42"), 42),
        _check("safe_int abc", lambda: safe_int("abc"), None),
        _check("safe_int 3.5", lambda: safe_int("3.5"), None),
        _check("parse_positive_int 7", lambda: parse_positive_int("7"), 7),
        _check_raises("parse_positive_int 0", lambda: parse_positive_int("0"), ValueError),
        _check_raises("parse_positive_int abc", lambda: parse_positive_int("abc"), ValueError),
    ]

    with TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing.txt"
        results.append(_check("missing file", lambda: describe_file_error(missing), "missing"))

        bad = Path(tmp) / "bad.txt"
        bad.write_bytes(b"\xff\xfe\x00")
        results.append(_check("bad encoding", lambda: describe_file_error(bad), "encoding"))

        good = Path(tmp) / "good.txt"
        good.write_text("apple banana\n", encoding="utf-8")
        results.append(_check("good file", lambda: describe_file_error(good), "ok"))

    passed = sum(results)
    print(f"\n{passed}/{len(results)} 通过")
    return passed == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
