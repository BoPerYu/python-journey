# w8-d2 · Python TF-IDF 参考检索

本目录提供一个不依赖 sklearn 的独立 Python 参考实现，用于验证
`w8-inverted-index` 的 C++ TF-IDF + 余弦检索。

## 定位

这是跨语言对拍工具，不是 Qwen 表示学习研究结果。

## 固定契约

- 一行一篇文档；
- 行内 token 使用空白分隔；
- 空行保留为空文档；
- `df` 是包含某词的文档数，不是该词的总词频；
- `idf = ln((N + 1) / (df + 1)) + 1`；
- 文档权重为 `tf * idf`；
- 分数为查询向量与文档向量的余弦；
- 分数降序，同分时 doc id 升序。

## 自测

```powershell
cd D:\Code\python-journey\w8-d2
python -m unittest -v .\test_reference_search.py
```

当前 8 项测试全部通过。

## 批量输出

```powershell
python .\batch_reference.py `
  --corpus D:\Code\cpp-journey\w8\w8-inverted-index\data\sample.tokens.txt `
  --queries D:\Code\cpp-journey\w8\w8-inverted-index\data\parity_queries.txt `
  --top-k 10 `
  --output .\artifacts\python_tfidf_small.tsv
```

## 已完成的 100,000 篇对拍

```text
查询数：10
每查询 Top-K：10
比较行数：100
文档 ID 顺序：完全一致
分数容差：abs(diff) <= 1e-9
结果：PASS
```

这些结果只证明两套实现遵守同一契约，不证明检索模型优于其他模型。
