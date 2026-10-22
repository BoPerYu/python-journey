r"""检查当前 Python 是否真的运行在项目虚拟环境里。

用法（在 D:\\Code\\python-journey\\w5-d1 目录下）：
    .\.venv\Scripts\python.exe venv_check.py
"""

import sys
import sysconfig


def main():
    in_venv = sys.prefix != sys.base_prefix
    print(f"Python:          {sys.version.split()[0]}")
    print(f"executable:      {sys.executable}")
    print(f"sys.prefix:      {sys.prefix}")
    print(f"sys.base_prefix: {sys.base_prefix}")
    print(f"in virtualenv:   {in_venv}")
    print(f"site-packages:   {sysconfig.get_paths()['purelib']}")

    if not in_venv:
        print("\n[FAIL] 当前不是虚拟环境；请运行 .\\.venv\\Scripts\\python.exe venv_check.py")
        return 1

    print("\n[OK] 当前 Python 来自项目虚拟环境")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
