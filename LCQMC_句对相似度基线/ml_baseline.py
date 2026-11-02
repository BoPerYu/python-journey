from pathlib import Path
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


DATA_DIR = Path(r"D:\Desktop\校\计算机\datasets\LCQMC")
TRAIN_PATH = DATA_DIR / "train.txt"
DEV_PATH = DATA_DIR / "dev.txt"


def load_pairs(path, limit=None, seed=42):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError("字段数错误")
            s1, s2, label_text = parts
            try:
                label=int(label_text)
            except ValueError:
               raise ValueError("label 不能转整数")
            if label not in (0,1):
                raise ValueError("label 不是0/1")
            rows.append((s1, s2, int(label)))
    if limit is not None and len(rows) > limit:
        rows = random.Random(seed).sample(rows, limit)
    return rows


def make_text(s1, s2):
    return f"{s1} [SEP] {s2}"


def main():
    train = load_pairs(TRAIN_PATH)
    dev = load_pairs(DEV_PATH)

    train_texts = []
    train_labels = []
    dev_texts = []
    dev_labels = []
    for s1, s2, label in train:
        text = make_text(s1, s2)
        train_texts.append(text)
        train_labels.append(label)
    for s1,s2,label in dev:
        text=make_text(s1,s2)
        dev_texts.append(text)
        dev_labels.append(label)

    # print("train pairs:", len(train))
    # print("train texts:", len(train_texts))
    # print("train labels:", len(train_labels))
    # print("dev pairs:", len(dev))
    # print("dev texts:", len(dev_texts))
    # print("dev labels:", len(dev_labels))
    # print("first train text:", train_texts[0])
    # print("first train label:", train_labels[0])
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
        )
    train_features=vectorizer.fit_transform(train_texts)
    dev_features=vectorizer.transform(dev_texts)

    model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    )
    model.fit(train_features,train_labels)
    predictions = model.predict(dev_features)
    accuracy=accuracy_score(dev_labels, predictions)
    macro_f1=f1_score(dev_labels, predictions, average="macro")
    probabilities = model.predict_proba(dev_features)[:, 1]

    print("train size:", len(train))
    print("dev size:", len(dev))
    print("accuracy:", accuracy)
    print("macro_f1:", macro_f1)

    errors = []
    for index, (s1, s2, true_label) in enumerate(dev):
        predicted_label = int(predictions[index])
        if predicted_label != true_label:
            errors.append(
                (
                s1,
                s2,
                true_label,
                predicted_label,
                probabilities[index],
                )
            )
    false_positives = []
    false_negatives = []
    confident_errors = []
    for x in errors:
        true_label = x[2]
        predicted_label = x[3]
        prob = x[4]
        if true_label == 0 and predicted_label == 1:
            false_positives.append(x)
        elif true_label == 1 and predicted_label == 0:
            false_negatives.append(x)
        confidence = prob if predicted_label == 1 else 1 - prob
        if confidence >= 0.8:
            confident_errors.append(x)
    print("total errors:", len(errors))
    print("false positives:", len(false_positives))
    print("false negatives:", len(false_negatives))
    print("confident errors:", len(confident_errors))

    selected = []
    selected.extend(false_positives[:1])
    selected.extend(false_negatives[:1])
    selected.extend(confident_errors[:1])
    for s1, s2, true_label, predicted_label, prob in selected:
        print()
        print("s1:", s1)
        print("s2:", s2)
        print("true:", true_label)
        print("pred:", predicted_label)
        print("prob:", prob)

if __name__ == "__main__":
    main()
