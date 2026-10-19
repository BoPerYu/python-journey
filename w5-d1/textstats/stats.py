"""负责“token 列表 -> 词频字典 -> Top-K”。"""


def count_tokens(tokens):
    """统计每个 token 出现次数，返回 dict[str, int]。

    例：["apple", "banana", "apple"] -> {"apple": 2, "banana": 1}

    提示：可以用 dict.get(word, 0) + 1，也可以用 collections.Counter。
    先用最朴素的方式写，别急着 import。
    """
    # TODO 2：实现这个函数，然后删掉下面的 raise
    a={}
    for x in tokens:
        a[x]=a.get(x,0)+1
    return a


def top_k(counts, k):
    """按“频次降序、同频单词升序”返回前 k 个 (word, count)。

    例：{"apple": 2, "banana": 1} -> [("apple", 2), ("banana", 1)]

    提示：
    - 先把 counts.items() 变成列表
    - 用 sorted(..., key=...) 或 list.sort(...)
    - 比较器里先看 count，count 相同再看 word
    - k 大于不同词数时，返回全部，不要报错
    """
    # TODO 3：实现这个函数，然后删掉下面的 raise
    c=sorted(counts.items(),key=lambda item: (-item[1], item[0]))
    # answer=[]
    # if len(c)>=k:
    #     for i in range(k):
    #         answer.append(c[i])
    # else:
    #     for i in range(c.len()):
    #         answer.append(c[i])
    # return answer
    return c[:k]