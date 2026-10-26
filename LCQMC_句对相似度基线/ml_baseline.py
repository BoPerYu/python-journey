r"""LCQMC 句对相似度最小基线（骨架）。

运行：
    .\.venv\Scripts\python.exe -X utf8 ml_baseline.py
"""

from pathlib import Path
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


DATA_DIR = Path(r"D:\Desktop\校\计算机\datasets\LCQMC")
TRAIN_PATH = DATA_DIR / "train.txt"
DEV_PATH = DATA_DIR / "dev.txt"


def load_pairs(path, limit=None, seed=42):
    """读取 LCQMC 的 s1, s2, label；limit 不为空时随机抽样。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            s1, s2, label = parts
            rows.append((s1, s2, int(label)))
    if limit is not None and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return rows


def make_text(s1, s2):
    return f"{s1} [SEP] {s2}"


def main():
    train = load_pairs(TRAIN_PATH, limit=20000)
    dev = load_pairs(DEV_PATH)

    # TODO 1：构造 train/dev 的 text 和 label 列表。
    train_texts = []
    train_labels = []
    dev_texts = []
    dev_labels = []

    # TODO 2：建立 TfidfVectorizer：
    # analyzer="char", ngram_range=(1, 2),
    # max_features=50000, min_df=2
    vectorizer = None

    # TODO 3：拟合 vectorizer，训练 LogisticRegression，
    # 在 dev 上预测并计算 accuracy / macro-F1。
    predictions = []
    labels = []

    # TODO 4：打印指标，并找 3 个错例。
    print("train size:", len(train))
    print("dev size:", len(dev))
    print("accuracy:", accuracy_score(labels, predictions) if labels else "TODO")
    print("macro_f1:", f1_score(labels, predictions, average="macro") if labels else "TODO")


if __name__ == "__main__":
    main()
