# W9 MacBERT：一批句对的一次训练更新

**11/20 复核**：本人核心已完成，22项外围检查通过；当前源码在 `artifacts/verified_20261120_review/training_check.json` 再次完成一次真实更新，前向、反向、step 各1次。下文填写说明保留为原任务参考；本次复核没有改写本人核心。完整笔记批注见 [当日复核记录](D:/Desktop/校/计算机/打卡训练/W9/W9D5_完成检查与笔记反馈_2026-11-20.md)。

本实验先检验数据、预训练主干、分类头、损失和优化器能否接通。用户填写一个函数，外围工具记录每一步的证据。用的是现成 MacBERT，不需要重写 Transformer。

## 1. 当前要打开哪个文件

打开 `D:\Code\python-journey\w9-macbert\student_step.py`，填写唯一的 `train_one_step` 函数。文件中有完整的输入、输出、边界、API 查阅提示和执行顺序。

其余文件暂时只在需要时看：

| 文件                        | 谁负责、做什么                                       |
| --------------------------- | ---------------------------------------------------- |
| `student_step.py`           | 本人串起前向、反向、一次更新                         |
| `experiment.py`             | AI 补齐数据读取、模型加载、观测、计时、结果保存      |
| `test_support.py`           | AI 验证外围会拒绝坏数据、缺失梯度和没有发生的更新    |
| `prepare_model.py`          | 固定官方版本，下载本实验需要的模型文件               |
| `data/sample_manifest.json` | 固定 4 对 train 样本，两类各两对，保存来源行号与指纹 |
| `artifacts/`                | 本地运行结果，不进入 Git                             |

## 2. 先分清三个名字

- **Python** 是你编写本实验的语言。
- **PyTorch**（代码中导入名为 `torch`）是计算张量、记录梯度关系、自动求梯度和更新参数的工具库。
- **Transformers** 是提供现成 MacBERT/BERT 结构、Tokenizer 和权重加载功能的库；本实验使用它的 PyTorch 模型。

你原来的 Python 函数、字典、变量和调用知识照常使用。先学本函数用到的 API，不要求先完成 PyTorch 全套课程。

可以把张量暂时理解成“带形状和数值类型的多维数组”。与普通 Python 列表相比，它还能参与库支持的高效运算和梯度计算。

## 3. 资源与本地加载

用户的 `D:\Code\BERT\config.json` 是 **24 层、1024 维**的 large 规格，原样保留。本实验改用独立目录：

```text
D:\Code\python-journey\models\chinese-macbert-base
```

官方来源为 `hfl/chinese-macbert-base`，固定版本：

```text
a986e004d2a7f2a1c2f5a3edef4e20604a974ed1
```

只需要配置、Tokenizer 文件和一份 PyTorch 权重，不需要同时下载 TensorFlow、Flax 的权重。`download_manifest.json` 保存文件大小、SHA-256 和来源。实验加载时只读本地，不自动联网，也不执行远程项目代码。

“浏览器下载”和“Python 下载”都是把文件保存到电脑。代码中写模型名称可以触发下载；代码中写本地目录则可以直接加载已经准备好的文件。本实验采用后者。

预训练权重尚未训练 LCQMC 二分类头。首次加载时，提示新分类头（有时包括 pooler）需要初始化是正常现象。编码器主干缺失权重会被外围拒绝。

## 4. 一批数据的含义

首轮使用清单中的前两对，batch size 为 2，最大序列长度为 64。4 对训练样本不代表正式训练规模。固定选两类各两条仅为检查流程，不能凭这点数据评价模型好坏。

| 字段             | 形状    | 含义                                    |
| ---------------- | ------- | --------------------------------------- |
| `input_ids`      | `[B,L]` | 词表中的整数编号                        |
| `attention_mask` | `[B,L]` | 哪些位置是真实输入，哪些为补齐          |
| `token_type_ids` | `[B,L]` | 哪些位置属于句段 A 或 B                 |
| `labels`         | `[B]`   | 每个句对的 0/1 标签，与句段编号含义不同 |

这里 `B=2`，`L` 是本批补齐后的 token 数，并不一定等于上限 64。各输入均为 CPU 上的 `torch.long` 张量。

`batch` 是保存上述字段的字典。`model(**batch)` 中的 `**` 表示按字典中的键名展开成关键字参数。它相当于把对应的输入分别交给模型，实际字段由外围已经准备好。

模型返回的 `outputs` 里：

- `outputs.logits` 是 `[B,2]`，每个句对有两个类别的原始分数。
- `outputs.loss` 是一个标量张量，由分数与 `labels` 计算得到，仍保留求梯度所需的计算关系。

库已经把分类头和分类损失接好，不需要本人再手动 softmax 或手写交叉熵。

## 5. 本人填写的职责与边界

函数签名是 `train_one_step(model, batch, optimizer, observe)`。`model`、`batch`、`optimizer` 都已准备好，`observe` 是外围递给你的观察工具。

本人根据函数内部注释串起数据流，调用三次 `observe`，分别标明前向完成、反向完成、更新完成。最后返回同一次前向的 `outputs`。预计是十行左右的核心连接代码，行数不是验收要求。

先对照这些用途思考再填写：

| API                                     | 改变什么                           |
| --------------------------------------- | ---------------------------------- |
| `model.train()`                         | 切换训练模式，本身不执行前向或更新 |
| `optimizer.zero_grad(set_to_none=True)` | 清理旧梯度                         |
| `model(**batch)`                        | 计算本次分数与损失                 |
| 张量的 `backward()`                     | 计算梯度                           |
| `optimizer.step()`                      | 用梯度执行参数更新                 |

不变量：一次前向、一次反向、一次更新；样本和标签不改动；主干与头都参与；不能把 loss 转成普通数字后再反向；不能用 `detach()` 切断训练关系；不要把整个训练函数包进 `no_grad()`。

`observe` 只观测和检查，不会替你运行反向传播或更新。AI 没有提供隐藏的参考训练实现。

## 6. 可直接复制的 PowerShell 命令

以下命令可从任何目录执行；只有外围测试要求先进入项目目录。

**准备样本**（已完成后重复运行会核对一致性）：

```powershell
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 'D:\Code\python-journey\w9-macbert\experiment.py' --stage prepare
```

预期关键输出：`SAMPLES_READY count=4 labels=2+2`。

**预检模型与前向**（这一步不训练）：

```powershell
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 'D:\Code\python-journey\w9-macbert\experiment.py' --stage preflight
```

预期关键输出：`PREFLIGHT_OK`、`no_parameter_update=True`，并生成 `artifacts\preflight.json`。不应出现 `UPDATE_VERIFIED`。

**本人填写后，执行一次真实更新**：

```powershell
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 'D:\Code\python-journey\w9-macbert\experiment.py' --stage train
```

未填写时预期输出 `STUDENT_TODO`，退出码为 2，并且不会开始加载训练模型。填写正确后预期输出 `UPDATE_OK step=1` 与 `UPDATE_VERIFIED`，生成 `artifacts\training_check.json`。这才是训练更新已验通的证据。

**一次更新通过后再测时**（首步热身，后五步统计中位数）：

```powershell
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 'D:\Code\python-journey\w9-macbert\experiment.py' --stage train --steps 6
```

此命令会继续在同一个小 batch 上更新六次，不能据此评价泛化效果。统计结果在 `median_seconds_excluding_first`。下载、导入、模型加载时间不混入训练步。该测时只对应当前 batch 和长度，不能直接当作论文配置耗时。

**检查外围工具**：

```powershell
Set-Location -LiteralPath 'D:\Code\python-journey\w9-macbert'
& 'D:\Code\python-journey\.venv-qwen313\Scripts\python.exe' -X utf8 -m unittest -v test_support
```

测试文件就在本目录的 `test_support.py`，预期末尾为 `OK`。测试使用微小线性层与异常观测信号，不调用学生训练函数，也不是一次 MacBERT 真实训练。2026-11-19 已通过 19 项外围检查。

## 7. 不通过时先检查哪里

| 现象              | 本人先检查的位置                                         |
| ----------------- | -------------------------------------------------------- |
| `STUDENT_TODO`    | 是否填写并删除了函数末尾的 `raise NotImplementedError`   |
| 观测顺序错误      | 三次 observe 的阶段名和所在步骤                          |
| loss 没有梯度关系 | 是否 detach、取成 `.item()`，或错误关闭梯度              |
| 缺少梯度          | 是否执行了反向；是否冻结了主干                           |
| 更新前参数已变化  | 是否在指定边界之前执行了更新或直接修改参数               |
| 更新后参数未变化  | 是否执行优化器更新；先看失败日志指出的是哪组参数         |
| logits 形状不符   | 是否保留二分类模型和原始 batch，是否返回了同一次 outputs |

训练失败时，已捕获的观测写入 `artifacts\training_failure.json`。外围可告诉你在哪个环节失败，关键原因先由本人尝试解释和定位。

## 8. 今天到哪里结束

今天只完成一个训练函数，并解释它如何使用输入、标签、loss、梯度和优化器。先跑一次验收，再决定是否测时。先不加冻结对照、不扩数据、不继续筛论文。

本实验使用当前 PyTorch 实现及极小样本，属于 MacBERT 的下游学习实验。它尚不是原论文结果复现，也不检验模型是否达到约 87% 或 88%。原始 test 不参与这次试验。

2026-11-19 已执行 `prepare` 和 `preflight`。固定首批张量为 `[2,34]`、标签为 `[2]`，未截断，输出为 `[2,2]`，被观测参数保持一致。真实的 MacBERT 训练更新和训练测时等待本人填写核心后执行。观测工具还记录实际前向、本次 loss 反向传播和优化器更新的调用次数，防止只有人工参数变化却被误认为一次正常训练。训练总耗时包含验证开销，各阶段时间另列。

官方依据：[MacBERT 资源](https://github.com/ymcui/MacBERT)、[BERT 分类接口](https://huggingface.co/docs/transformers/model_doc/bert)、[PyTorch 自动微分](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

## 9. 本人填写后的首次验收（2026-11-19）

已按本人实际填写的 `student_step.py` 执行 `--stage train --steps 1`，核心顺序正确，未改本人训练主体。独立一行 `outputs.loss` 没有额外作用，`return` 后的模板异常不可达，可由本人日后清理。外围原先按源码字符串查 TODO 会误拦这份已完成的函数，现已修正为识别顺序骨架中可达的模板异常，并增加对应检查。

- 真实更新结果：`UPDATE_VERIFIED`，前向、反向、优化器更新计数均为 1。
- 单步合计 1.604249 秒，含第一次运行与观测开销，不是稳定训练速度。
- 阶段计时约为前向 0.442 秒、反向 0.508 秒、优化器更新 0.618 秒。阶段合计与单步合计的差额含观测检查等外围开销。
- 模型加载 1.435862 秒；整个 Python 程序从启动到结束为 110.640 秒，还包含导入、校验等，不能将它视为每一步训练成本。
- 分类头、主干第 1 层 Q 投影、第 12 层 Q 投影的梯度范数分别约为 6.008506、0.598266、0.293938，均为有限非零值。
- 上述三组被观测权重在反向后、更新前的最大绝对变化均为 0；更新后分别约为 0.0000200011、0.0000200272、0.0000200011。
- 完整外围检查已通过 22 项；真实日志为 `artifacts/training_check.json`，进程总计时及日志指纹为 `artifacts/process_timing.json`。

`batch` 是字典，一行对应一个完整句对，不是一行对应一个句子。本次两对样本分别有 34、21 个有效 token，第二对补 13 个 `[PAD]`，因此前三个字段均为 `[2,34]`。`labels=[1,0]` 每对只有一个标签，所以形状为 `[2]`。标签用于计算损失，不被当成输入 token 提供给主干。

`observe` 是 `StepObserver` 的可调用对象。创建时保存模型和选定参数的独立副本；之后三次调用分别检查阶段顺序、输出形状、梯度和参数差异。它还通过 hook（事件回调）记录给定模型的前向、本次 loss 的反向以及优化器更新次数。观测器读取数据，不代替学生运行训练。

三个阶段日志中的 loss 都是 0.5160534382，因为它们引用同一次前向得到的损失；后两次观察没有重新前向计算更新后的 loss。这不是参数未变化的证据，也不应据此评价分类效果。
