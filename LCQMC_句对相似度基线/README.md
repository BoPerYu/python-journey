# LCQMC 句对相似度基线

这是一个**独立于 `w5-d1` 的额外项目**，不占用 W6 核心任务时间。

## 项目目标

输入一对中文句子，判断它们语义是否相近：

```text
s1<TAB>s2<TAB>label
label = 1：相近
label = 0：不相近
```

第一版只做一个最小监督分类闭环：

```text
句对 -> 字符 n-gram TF-IDF -> 逻辑回归
     -> dev accuracy / macro-F1
     -> 3 个错例
```

这不是完整检索系统，也不是论文复现。它的作用是：

- 执行老师“先自学机器学习、边做边补”的建议；
- 把 Python、文本处理和 W7 的检索项目连接起来；
- 形成一个能解释、能复现的最小基线。

## 数据

使用本地数据：

```text
D:\Desktop\校\计算机\datasets\LCQMC\
├── train.txt
├── dev.txt
└── test.txt
```

当前公开副本的三份文件都含 `label` 列；训练只使用 train 的 2 万条抽样，dev 全部用于评估，不把 dev 放进训练。

## 环境

建议在本目录创建自己的虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前骨架也可以在 `..\w5-d1\.venv` 中运行。

## 当前状态

`ml_baseline.py` 是骨架，已完成数据读取和抽样，尚未训练模型。

必须先完成“能够解释”的阶段，再填模型：

1. 标签、特征、train/dev；
2. TF-IDF：字符 1–2gram、`max_features=50000`、`min_df=2`；
3. 逻辑回归；
4. accuracy、macro-F1、错例分析。

## 运行

```powershell
cd D:\Code\python-journey\LCQMC_句对相似度基线
..\w5-d1\.venv\Scripts\python.exe -X utf8 ml_baseline.py
```

当前预期输出包括：

```text
train size: 20000
dev size: 8802
accuracy: TODO
macro_f1: TODO
```

## 联系老师的门槛

只有能做到以下事情，才把这个项目用于汇报：

- 能从空白解释数据流；
- 能解释 `[SEP]`、TF-IDF、逻辑回归、train/dev；
- 能解释至少一个错例；
- 代码中每一行都能讲清；
- 不把 AI 生成内容说成自己独立完成；
- README 记录真实命令、数据路径、指标和限制。

如果 80% 是复制且讲不清，就不发给老师，也不写进简历。

## 与 W6 的关系

本项目是**额外项目**。W6 的 C++ 工程化、Python 数据侦察/类、数据结构作业、LeetCode 和官方 ML 自学一项都不能因此缩减。
