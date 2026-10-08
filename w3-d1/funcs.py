scores = [89, 91, 70, 78, 74, 94, 59, 81]                 # W2D2 用过的那 8 个
scores2 = [89, 91, 70, 78, 74, 94, 59, 81, 91, 89]         # 集合那天的 10 个
people = {"张三": 89, "李四": 91, "王五": 70, "赵六": 78, "钱七": 94}

def total_and_average(scores):
    total=0
    average=0
    for score in scores:
        total+=score
    average=total/len(scores)
    return total,average
# —— 返回**总分和平均分**两个值
def grade(score):
    if score>=90:
        return "A"
    if score>=80:
        return "B"
    if score>=70:
        return "C"
    if score>=60:
        return "D"
    else:
        return "F"
# —— 返回等级字符串（≥90 A，≥80 B，≥70 C，≥60 D，否则 F）
def best(people):
    best_name=None
    best_mark=0
    for name,mark in people.items():
        if best_mark<mark:
            best_mark=mark
            best_name=name
    return best_name,best_mark
# —— 返回**分数最高的人的名字和分数**两个值
def unique_count(scores2):
    return len(set(scores2))
# —— 返回"有几个不同的数"（用你学过的 set）
def safe_get(people, key, default=0):
    return people.get(key,default)
# —— 查得到就返回值、查不到返回 `default`
print(total_and_average(scores))
print(grade(90))
print(best(people))
print(unique_count(scores2))
print(safe_get(people,"小王"))

