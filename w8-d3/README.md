# w8-d3 · Qwen 与 TF-IDF 同口径实验

主文件：

```text
qwen_compare.py
```

该脚本负责编排完整实验，但不复制以下已有职责：

- `w8-d1/sample_pairs.py`：原始 TSV 读取、canonical 分组、泄漏审计和固定 dev 抽样；
- `w8-d2/reference_search.py`：TF-IDF 检索参考实现。

## 当前状态

完整实验流程已实现并通过 50 对和 500 对试点。

2026-11-16 固定 seed=42 的 500 对结果：

| 模型 | Accuracy | Macro-F1 | ROC-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen + threshold | 0.7720 | 0.7720 | 0.8798 | 87 | 27 |
| TF-IDF + LR | 0.5900 | 0.5806 | 0.5974 | 93 | 112 |

Qwen 阈值只在 2,000 条 train threshold 样本上选择，固定为
`0.824390411377`。dev 500 只用于最终评估，test 未使用。

## 计划运行环境

```text
D:\Code\python-journey\.venv-qwen313\Scripts\python.exe
```

该环境已验证包含：

```text
Python 3.13.14
sentence-transformers 6.2.0
scikit-learn 1.9.1
```

## 实现顺序

1. `build_clean_train_pool`
2. `build_fit_and_threshold_pools`
3. `assert_disjoint`
4. `collect_unique_texts`
5. `load_qwen_model`
6. `encode_unique_texts`
7. `compute_pair_cosines`
8. `predict_at_threshold`
9. `compute_metrics`
10. `select_threshold`
11. TF-IDF 训练与预测
12. CSV、JSON、README 输出

## 运行

50 对冒烟：

```powershell
& .\.venv-qwen313\Scripts\python.exe .\w8-d3\qwen_compare.py `
  --dev-limit 50 `
  --output-dir .\w8-d3\artifacts\dev50
```

500 对正式试点：

```powershell
& .\.venv-qwen313\Scripts\python.exe .\w8-d3\qwen_compare.py `
  --dev-limit 500 `
  --output-dir .\w8-d3\artifacts\dev500
```

## 输出

- `qwen_dev_scores.csv`：Qwen 分数、标签和预测；
- `tfidf_dev_scores.csv`：TF-IDF 正类概率、标签和预测；
- `metrics.json`：两模型指标与数据哈希；
- `comparison.csv`：两模型指标对比；
- `errors_qwen.json`：Qwen FP/FN；
- `errors_tfidf.json`：TF-IDF FP/FN；
- `README.md`：本次实验口径和结果。

## 解释边界

这是两条端到端流程的比较：

```text
TF-IDF + 有监督逻辑回归
vs
Qwen 句向量 + 无监督余弦阈值
```

因此结果不能单独归因于“Qwen 的表示能力更强”。当前 500 对是固定小样本
试点，不是完整 dev，也不是泛化结论。
