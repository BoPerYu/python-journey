# w6-d1 · LCQMC 数据侦察

本目录只做数据检查，不训练模型。脚本用 Python 标准库逐行读取 TSV，
统计 UTF-8 编码、行数、标签分布和句长，不把 16 MB 级文本一次性交给 pandas。

本机这份数据的 `train.txt`、`dev.txt`、`test.txt` 都是三列
`s1<TAB>s2<TAB>label`；脚本会逐份验证，不根据文件名猜测列数。

## 运行

先把 LCQMC 三方数据放在：

```text
D:\Desktop\校\计算机\datasets\LCQMC\
├── train.txt
├── dev.txt
└── test.txt
```

然后运行：

```powershell
cd D:\Code\python-journey\w6-d1
..\w5-d1\.venv\Scripts\python.exe -X utf8 data_recon.py
```

可用 `--data-dir` 指向别的目录：

```powershell
..\w5-d1\.venv\Scripts\python.exe -X utf8 data_recon.py --data-dir D:\path\to\LCQMC
```

## 统计口径

- `records`：成功解析的数据行数，不含表头。
- `physical_lines`：文本中的物理行数，包含异常行。
- `sentence_chars`：把一对句子的 `s1` 和 `s2` 都看作样本后的字符数；
  一个字符按一个 Unicode code point 计，不等同于中文词数或 UTF-8 字节数。
- `s1_chars` / `s2_chars`：分别统计两侧句子的字符数。
- `p50` / `p90` / `p95` / `p99`：线性插值百分位数。
- 标签比例的分母是成功解析的记录数。
- `overlap_audit`：检查同一句对在集合内重复、跨 train/dev/test 重叠，
  以及同一句对在不同集合中标签是否冲突。它是数据泄漏风险检查，
  不是模型指标。

## 为什么先侦察

在划分 train/dev、构造 TF-IDF 和计算指标之前，必须先回答：

1. 文件能否按 UTF-8 严格解码？
2. 字段数与标签值是否符合约定？
3. 两类样本是否失衡？
4. 句子长度是否存在极端值，可能拖慢向量化和模型训练？

脚本只报告观察结果，不替数据做过滤或标签修正。
