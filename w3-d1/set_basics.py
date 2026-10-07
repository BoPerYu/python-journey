scores = [89, 91, 70, 78, 74, 94, 59, 81, 91, 89]
a班 = {"张三", "李四", "王五", "赵六"}
b班 = {"王五", "赵六", "钱七", "周八"}
text = "数据科学与大数据技术"
print(len(scores))
print(len(scores)-len(set(scores)))
print(set(scores))
print(scores)
print(91 in scores)
print(60 in scores)
print(a班 & b班)
print(a班 | b班)
print(a班 - b班)
print(len(a班 & b班))
print(len(a班 | b班))
print(len(a班 - b班))
s=set(text)
print(len(s))
print(s)