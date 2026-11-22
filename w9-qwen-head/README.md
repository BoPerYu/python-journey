# 11/20 Python：冻结 Qwen，训练一个句对分类头

**11/20 完成验收**：本人已完成 `student_head.py` 两个核心。33项检查通过（学生固定用例4项、外围29项）；在 `artifacts/verified_20261120_review` 独立拟合，权重、偏置与预测均与原产物一致。原结果未覆盖。以下各节保留任务说明与复跑方法，不表示核心仍为 TODO。本人笔记批注与结果解释的验收状态见 [当日复核记录](D:/Desktop/校/计算机/打卡训练/W9/W9D5_完成检查与笔记反馈_2026-11-20.md)。

## 今天的唯一主要目标

用现有 Qwen 把句子变成固定向量，自己实现“把两个句向量变成一行句对特征”，再用 train 标签拟合逻辑回归分类头。本日只有 20 对 train，以及原 dev500 中的 20 对诊断样本，先把流程接通。

原始 test 不参与。dev20 是已经使用过的固定 dev500 的子集，不是完整 dev 或独立测试集，不凭这个小结果判断方法优劣。

## 1. 昨天与今天的区别

昨天 MacBERT：两句联合编码，分类损失推动主干与分类头更新。

今天 Qwen：两句分别编码，Qwen 参数冻结；句向量先保存到磁盘，之后训练分类头只读取这些向量。

```text
左句 → 固定的 Qwen → 左向量 ┐
                           ├→ 本人构造句对特征 → 分类头 → 匹配预测
右句 → 固定的 Qwen → 右向量 ┘                      ↑
                                             train 标签用于拟合
```

这里的分类头使用已经学过的 sklearn LogisticRegression。`.fit()` 内部负责拟合权重，本人不需要再写 PyTorch 的 backward/step。前向、梯度、优化器这些概念仍然成立，只是这次由库封装执行。分类头训练时不运行 Qwen，因而不会更新 Qwen 参数。

## 2. 本人只填写一个文件的两处

打开 `D:\Code\python-journey\w9-qwen-head\student_head.py`。

### 任务 A：make_pair_features

输入 `left`、`right` 是两个形状相同的二维 NumPy 数组，均为 `[N,D]`。

- N 是句对数。第 i 行的左向量与第 i 行的右向量属于同一对。
- D 是每句向量维度，本地 Qwen 为 1024。
- 输出 `[N,2D]`。每一行前半部分是对应维度的绝对差，后半部分是对应维度的乘积。

这两部分保留了两句在不同维度上的关系，让分类头学习如何组合它们。单独某一维没有预先规定的语义，也不能把乘积的正负直接解释成句意匹配或不匹配。

先看一个缩成两维的教学例子：

| 内容 | 数值 |
| --- | --- |
| 左向量 | `[1, -2]` |
| 右向量 | `[-1, 2]` |
| 逐维绝对差 | `[2, 4]` |
| 逐维乘积 | `[-1, -4]` |
| 横向拼接后的一行特征 | `[2, 4, -1, -4]` |

输入只有一对时是一行，两对时就有两行，不要把它们合并成一行。沿 `axis=1` 接列，会保留句对的行数。可查阅 `np.abs`、数组逐元素乘法、`np.concatenate` 的作用，具体组合由本人实现。

不变量：交换两句，特征保持不变；不修改原始向量；不改变句对行顺序；不使用标签计算特征。

### 任务 B：fit_head

外围已配置好 `classifier`，你拿到的是：

| 对象 | 本次形状或含义 |
| --- | --- |
| `train_features` | `[20,2048]`，20 个句对，每对一行 |
| `train_labels` | `[20]`，与特征逐行对应的标签 |
| `classifier` | 已设置好参数、尚未 fit 的逻辑回归对象 |

本人将这组训练特征和训练标签交给 `.fit(X,y)`，返回同一个分类头。不要把 dev 标签用于拟合。这个步骤承接 W8 的 LR 使用经验，区别是输入特征从 TF-IDF 换成了你构造的 Qwen 句对特征。

## 3. 工程外围已准备什么

- train20 从 W8 原 threshold2000 中取两类各十对。原 threshold2000 仍按原始 clean train、fit_size=20000、threshold_size=2000、seed=42 的完整过程重建，不是重新随意抽样。
- dev20 是原固定 dev500 清单中的前20对，另存 ID 和原始行号，两句与标签核对原文件。
- 清单为 `data/smoke_manifest.json`。原始行号沿用 W8 从1开始，不能与 MacBERT 试验中的零起始 ID 数字直接混用。
- 对两组共80条不同句子统一编码，保存 `[80,1024]` 的 float32 向量缓存。两句都不加 query instruction，与 W8 的编码方式一致。
- CPU float32、最后 token pooling、向量 L2 归一化。编码长度上限设置512，并实际检查没有句子被截断；这是对原 W8 未显式设上限的补充记录。
- 生成缓存时 Qwen 设置为 eval，关闭参数梯度，使用 inference mode；编码前后比较全部参数指纹，并检查没有梯度。
- 修改本人函数后，训练分类头直接复用缓存，不重新运行 Qwen。

外围脚本是 `pipeline.py`；固定小例子的学生验收是 `test_student_head.py`；外围机制检查是 `test_support.py`。

## 4. 顺序与可复制命令

在 PowerShell 中先进入目录：

```powershell
Set-Location -LiteralPath 'D:\Code\python-journey\w9-qwen-head'
```

**第一步：读任务 A 的例子，填写 make_pair_features，然后运行固定用例。**

```powershell
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 -m unittest -v test_student_head
```

预期末尾是 `Ran 4 tests` 和 `OK`。没填写时会报 NotImplementedError，这是待完成标记，不是环境故障。

**第二步：填写 fit_head，运行整个小流程。**

```powershell
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 pipeline.py --stage fit
```

预期 `HEAD_SMOKE_OK train=20 dev_diagnostic=20 features=(20, 2048)`。两处函数未填完时会输出 `STUDENT_TODO`，不会训练分类头。

它会保存 `artifacts/head_report.json`、`dev20_predictions.csv` 和 `head.joblib`。同时检查左右交换预测一致、缓存未改变、模型保存后重载预测一致。报告里的指标先由本人解释，不把20对诊断当作方法强弱结论。

**若只想看数据形状和缓存是否准备好：**

```powershell
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 pipeline.py --stage inspect
```

预期 `CACHE_INSPECT_OK train_left=(20, 1024) dev_left=(20, 1024) train_labels=(20,)`。

**准备与检查工具的命令，仅在需要时使用：**

```powershell
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 pipeline.py --stage prepare
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 pipeline.py --stage encode
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 -m unittest -v test_support
```

prepare 预期 `SAMPLES_READY train=20 dev_diagnostic=20 unique_texts=80`。encode 首次生成缓存后显示 `CACHE_READY`，以后相同设置会复用缓存。最后一条只检查外围工具，不替你实现核心。

## 5. 没通过时本人先看什么

| 现象 | 先检查 |
| --- | --- |
| 输出变成40行，或只有一行 | 拼接方向、是否把两句或不同句对混到行轴 |
| 数字用例不符 | 绝对差和逐维乘积是否各自保留了全部维度、两部分次序是否正确 |
| 交换两句后特征不同 | 是否加入了单独左句/右句特征，或用到了依赖句序的逻辑 |
| 原向量被修改 | 是否使用了原地操作 |
| 分类头尚未拟合 | fit_head 是否真的传入 train_features 与 train_labels 并调用 fit |
| 缓存或数据指纹不符 | 先定位输入或文件变化，不删除清单强行覆盖 |

## 6. 今天的时间安排

Python / NLP 最多约130分钟，包含读讲义、查接口、编码和排错，并计入本周20小时上限：约20分钟理解这条数据流，约40分钟实现与测试特征，约25分钟接通分类头，约20分钟检查实际产物，约25分钟作为返工缓冲。不要求用满，也不因此挤占 C++、力扣或周末休息。

今日完成标准是：本人两处核心能运行，能说明哪些参数被训练、哪些保持固定，能从预测文件中找到一条句对及其标签和预测。先不扩到200/2000对，不新增注意力理论。

API 依据：[逻辑回归 fit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)、[SentenceTransformer 编码](https://sbert.net/docs/package_reference/sentence_transformer/model.html)。

## 7. 11/20 已实测的准备结果

`prepare` 与 `encode`、`inspect` 均已实际运行通过。共80条不同文本，缓存 `[80,1024]`，最长24个 token，无截断；Qwen 加载2.792秒、编码19.377秒（不含导入及额外校验等开销）。全部参数关闭梯度，编码前后参数指纹一致。

本人完成后，`fit` 已实际通过；原拟合调用约0.0666秒，独立复核约0.0194秒，二者均不含编码、文件校验等完整流程开销。分类头系数、偏置与20条预测完全相同，保存后重载与交换句序预测一致。

外围冻结记录已补强：指纹缺失、非64位十六进制或前后不同均拒绝。完整 `test_support test_student_head` 共33项通过。缓存读取继续核对 torch、NumPy、Sentence Transformers 版本及冻结记录；复核记录另保存当前依赖版本，未倒填原缓存的历史环境。
