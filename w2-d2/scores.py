import sys
sys.stdout.reconfigure(encoding="utf-8")
scores = [89, 91, 70, 78, 74, 94, 59, 81]
print(len(scores))
print(scores[0], scores[-1])
total=0
for score in scores:
    total+=score
average=total/len(scores)
print (f'总和是{total}，平均数是{average}')
for i in scores:
    if i>=90:
        print(f'{i}等第为A。')
    elif i>=80:
        print(f'{i}等第为B。')
    elif i>=70:
        print(f'{i}等第为C。')
    elif i>=60:
        print(f'{i}等第为D。')
    else:
        print(f'{i}等第为F。')
i=0
while i<len(scores):
    print(f'scores[{i}]={scores[i]}')
    i+=1
mx=scores[0]
mn=scores[0]
i=0
for i in range(0,len(scores)):
    if mx<scores[i]:
        mx=scores[i]
    if mn>scores[i]:
        mn=scores[i]
print(f'Max={mx},Min={mn}')