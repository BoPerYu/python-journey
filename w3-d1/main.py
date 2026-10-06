scores = {"张三": 89, "李四": 91, "王五": 70, "赵六": 78, "钱七": 94}
print(len(scores))
print(scores)
print(scores.keys())
print(scores["李四"])
print(scores.get("周八",0))
scores["周八"]=66
print(scores)
scores["王五"]=75
print(scores)
del scores["张三"]
print(scores)
total=0
average=0
for i in scores.values():
    total+=i
average=total/len(scores)
print(f'total={total},average={average}')
max_name=None
max_score=-1
for name,score in scores.items():
    if score>max_score:
        max_score=score
        max_name=name
print(f'Max={max_score}，name={max_name}')
