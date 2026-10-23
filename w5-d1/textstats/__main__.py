"""python -m textstats 的入口。"""

from . import count_tokens, read_tokens, top_k
import sys

def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    path = args[0] if args else "data/sample.txt"

    try:
        tokens = read_tokens(path)
    except FileNotFoundError:
        # TODO：打印到 stderr，返回 1
        print(f"找不到文件：{path}",file=sys.stderr)
        return 1
    except UnicodeDecodeError:
        # TODO：打印到 stderr，返回 1
        print(f"文件解编码不一致：{path}",file=sys.stderr)
        return 1
    except OSError as e:
        # TODO：打印到 stderr，返回 1
        print(f"读取文件失败{e}",file=sys.stderr)
        return 1

    counts = count_tokens(tokens)
    for word, count in top_k(counts, 10):
        print(f"{word} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())