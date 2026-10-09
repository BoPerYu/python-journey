scores = [89, 91, 70, 78, 74, 94, 59, 81]
with open("score.txt","w",encoding="utf-8")as f:
    for i in scores:
        f.write(f'{i}\n')
with open("score.txt","r",encoding="utf-8")as f:
    print(f.read())
    f.seek(0)
    for i,line in enumerate(f,1):
        line=line.strip()
        print(f'第{i}个是：{line}')

with open("score.txt","r",encoding="utf-8")as f:
    total=0
    num=0
    for line in f:
        total+=int(line)
        num+=1
    average=total/num
    print(total,average)

with open("score.txt", "a", encoding="utf-8") as f:
    f.write("100\n")

with open("score.txt","r",encoding="utf-8")as f:
    shu=0
    for line in f:
        shu+=1
    print(f'共有{shu}行')