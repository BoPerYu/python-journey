import sys
sys.stdout.reconfigure(encoding="utf-8")
name="Yuchenkai"
major="AI"
grade="大二"
print(f"{name}的专业是{major}，今年{grade}")

# ---------- 第 2 题：先猜，再跑 ----------
# 预期（补记）：按 C++ 的习惯，我以为三行会是  3  3  1
# 实际跑出来：                                3.5  3  1
# 对不上的是第一个：Python 里单个 "/" 永远是【小数除法】，
# 想要整除必须写两个斜杠 "//"，取余还是 "%"。
print(7/2,7//2,7%2)

s="Hello Python"
print(f"长度   = {      len(s)      }")
print(f"全大写 = {      s.upper()      }")
print(f"前五个 = {      s[0:5]      }")

# ---------- 第 4 题：故意写错，把报错抄下来 ----------
# 下面这行放开注释就会报错（现在留着不影响运行）：
# print(a)
#
# 跑出来的原文（Python 的 traceback 从下往上读）：
#   Traceback (most recent call last):
#     File "...\basics.py", line 11, in <module>
#       print(a)
#             ^
#   NameError: name 'a' is not defined
#
# 一句话解释：最后一行是"错误类型 + 原因"——名字 a 从来没有被定义过；
# 上面几行告诉你出错的文件、行号和那一行代码，^ 指到出问题的位置。
