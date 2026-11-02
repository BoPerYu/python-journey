# LCQMC 句对语义匹配最小基线

## 一、项目定位

输入一对中文句子，判断它们是否表达相近意思：
s1<TAB>s2<TAB>label
label = 1：语义相近
label = 0：语义不相近
这是一个有监督二分类基线。
当前目标是用真实中文数据完成一个可运行、可复现、可解释的最小闭环。

环境：
Python 3.14.6
scikit-learn 1.9.1
numpy 2.5.4
scipy 1.18.1

数据规模：
train：238,766 条
dev：8,802 条
test：12,500 条

数据读取会严格检查：

- 是否恰好有三列；
- label 是否可以转换为整数；
- label 是否只取 0 或 1。
  发现异常数据时直接报错，不静默跳过坏行。

## 二、当前数据流

train/dev TSV
-> 逐行读取并校验
-> (s1, s2, label)
-> s1 + " [SEP] " + s2
-> 字符 1-2gram TF-IDF
-> LogisticRegression
-> dev predict
-> accuracy / macro-F1
-> 错例分析

TF-IDF：
TfidfVectorizer(
analyzer="char",
ngram_range=(1, 2),
max_features=50000,
min_df=2,
)

分类器：
LogisticRegression(
max_iter=1000,
class_weight="balanced",
)

训练集使用 fit_transform：学习词表和 IDF，并转换训练文本。
dev 只使用 transform：沿用训练集学到的词表和 IDF，避免数据泄漏。

## 三、实验

实验 A：原定采样基线
train：从完整 train 中按 seed=42 随机抽取 20,000 条
dev：完整 8,802 条
accuracy：0.6292887980
macro-F1：0.6284866430
错误总数：3,263
假阳性：1,426
假阴性：1,837
高置信错误（预测类别置信度 >= 0.8）：265

实验 B：全量 train 变体
train：完整 238,766 条
dev：完整 8,802 条
accuracy：0.6951829130
macro-F1：0.6950883601
错误总数：2,683
假阳性：1,263
假阴性：1,420
高置信错误：509

## 四、错例分析

以下三条来自实验 A（20,000 条 train）：

1. 实体范围不同

   s1：这腰带是什么牌子
   s2：护腰带什么牌子好
   真实标签：0
   预测标签：1
   label=1 概率：0.6676

分析：
两句共享“腰带”“什么牌子”等字符，字符 n-gram 相似度高。
但“腰带”和“护腰带”指代的产品范围不同，字符共现没有准确表达实体边界。

2. 同义改写与语序变化

   s1：校花的贴身高手打给什么时候完结
   s2：校花的贴身高手什么时候可以写完
   真实标签：1
   预测标签：0
   label=1 概率：0.4407

分析：
两句都在询问作品何时完结，但“打给什么时候完结”和
“什么时候可以写完”表达方式不同。当前拼接后的字符袋特征缺少句内对齐，
对同义改写和语序变化建模不足。

3. 同义短语、数量与包装差异

   s1：一盒香烟不拆开能存放多久？
   s2：一条没拆封的香烟能存放多久。
   真实标签：1
   预测标签：0
   label=1 概率：0.1982

分析：
两句核心问题都是未拆封香烟能存放多久，但“一盒/一条”和
“拆开/拆封”带来字面和语序差异。模型错误地放大了这些差异。

## 五、运行

### 1. 安装依赖

    pip install -r requirements.txt

### 2. 准备数据

原始数据不随仓库分发，获取方式见文末“数据获取”。把 train.txt、dev.txt、
test.txt 放在同一个目录，然后用环境变量 LCQMC_DATA_DIR 指向该目录。

PowerShell：

    $env:LCQMC_DATA_DIR = "D:\path\to\LCQMC"

cmd：

    set LCQMC_DATA_DIR=D:\path\to\LCQMC

bash：

    export LCQMC_DATA_DIR=/path/to/LCQMC

不设置环境变量时，脚本会退回使用 ml_baseline.py 顶部的 DEFAULT_DATA_DIR。
该默认值只是作者本机的路径，换机器时请改掉或改用环境变量。

### 3. 运行

在仓库根目录执行：

    cd LCQMC_句对相似度基线
    python -X utf8 rule_baseline.py
    python -X utf8 ml_baseline.py

rule_baseline.py 不需要第三方依赖，也不需要数据文件，可以直接运行。

### 4. 切换实验 A / B

实验 A 使用：

    train = load_pairs(TRAIN_PATH, limit=20000)

实验 B 使用：

    train = load_pairs(TRAIN_PATH)

切换训练规模后必须重新运行完整脚本，不能把两个指标混写。

## 六、已知限制

- 当前只使用字符 n-gram，不能充分建模词义、否定、语序和远距离关系；
- s1 和 s2 被拼成一个文本，没有显式句对对齐；
- train 正类比例约为 58.04%，因此使用 class_weight="balanced"，并同时报告
  accuracy 与 macro-F1；
- train/dev 存在完全相同的句对，dev 指标可能因此略偏乐观；
- 完整 train 包含少量重复句对和标签冲突；
- dev 同时用于评估与错例分析；
- test 暂未用于训练、调参或最终报告；
- 当前结果不能证明对其他领域或线上文本有同等泛化能力。

## 七、AI 辅助说明

本项目在 AI 辅助下完成。AI 提供了项目骨架、API 说明、调试提示和部分
初始实现；本人逐步修改、运行并复核了数据读取、TF-IDF、逻辑回归、
指标计算和错例分析。不将 AI 生成内容声称为本人完全独立完成。

本人正在完成并复核：

- 数据读取与字段校验；
- TF-IDF 输入输出；
- train/dev 边界；
- 逻辑回归训练和预测；
- accuracy、macro-F1；
- 错例筛选与解释；
- 从空白复述完整数据流。

## 八、下一步

1. 在稳定留出的验证集上比较字符 1-2gram 与 1-3gram；
2. 比较“拼接文本”与“分别编码两句话”的差别；
3. 对重复句对和标签冲突做更严格的数据审计；
4. 固定方法后才使用 test；
5. 将实验配置、指标和限制持续写回本文件。

## 数据获取

原始数据不随本仓库分发，使用公开 LCQMC 数据：

- 来源：https://github.com/qhduan/CSTS
- 期望文件：train.txt、dev.txt、test.txt
- 每行格式：s1<TAB>s2<TAB>label
- 将三个文件放在同一目录，用环境变量 LCQMC_DATA_DIR 指向该目录；
  不设置时脚本使用 ml_baseline.py 顶部的 DEFAULT_DATA_DIR。

本机核验的 SHA-256：
train.txt
8D9D196CB8B86E53A6CCB9D2C0682844D973EEBE089590B2C46388A03D63F639

dev.txt
D3CD41ED89B8F6196A76EBFF2AA03D5ECF42EDCD12942AA865D34F7E570AF7C2

test.txt
8969F9C16050F40DF9DA6F591443A47BC14C17A504A354EE4CC4AB5B9AA303D9
