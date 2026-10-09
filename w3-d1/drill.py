nums = [10, 20, 30, 40, 50]
dups = [1, 2, 2, 3, 3, 3]
s = {10, 20, 30, 40, 50}
d = {"a": 1, "b": 2}
print(nums[2])
print(nums[len(nums)-1])
# 第二种
print(nums[-1])
for i in range(0,5):
    print(f'{i}')
for i in range(1,6):
    print(f'{i}')
for i,x in enumerate(nums):
    print(f'下标{i}:值{x}')
# print(s[2]),报错：TypeError: 'set' object is not subscriptable
print(30 in nums)
print(30 in s)
print(len(set(dups)))
a={1,2,3}
b={3,4}
print(a&b)
print(a|b)
print(a-b)
for i,x in d.items():
    print(f'键{i},值{x}')
for i in range(len(nums)-1,-1,-1):
    print(nums[i])
c=[3, 5, -2, 7]
for i in range(0,len(c)):
    if c[i]<0:
        print(f'{i}')
        break
d={"数","据","科","学"}
f={"大","数","据"}
print(d&f)
