from __future__ import annotations

import argparse
import json
import math
import random
import torch
from sentence_transformers import SentenceTransformer
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import csv

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


JOURNEY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNEY_ROOT / "w8-d1"))

from sample_pairs import (  # noqa: E402
    PairRecord,
    group_by_canonical,
    load_split,
    sha256_file,
)


DEFAULT_DATA_DIR = Path(r"D:\Desktop\校\计算机\datasets\LCQMC")
DEFAULT_MODEL_DIR = JOURNEY_ROOT / "models" / "Qwen3-Embedding-0.6B"
DEFAULT_MANIFEST = JOURNEY_ROOT / "w8-d1" / "artifacts" / "sample_manifest.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    manifest_path: Path
    output_dir: Path
    model_dir: Path
    fit_size: int
    threshold_size: int
    dev_limit: int
    batch_size: int
    seed: int


@dataclass(frozen=True)
class ScoredPair:
    record: PairRecord
    score: float


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_f1: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    threshold_metrics: Metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Qwen cosine thresholding with TF-IDF on LCQMC."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--fit-size", type=int, default=20_000)
    parser.add_argument("--threshold-size", type=int, default=2_000)
    parser.add_argument(
        "--dev-limit",
        type=int,
        default=50,
        help="50 for smoke; 500 for the formal pilot.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.fit_size <= 0:
        raise ValueError("--fit-size must be positive")
    if args.threshold_size <= 0:
        raise ValueError("--threshold-size must be positive")
    if args.dev_limit <= 0 or args.dev_limit > 500:
        raise ValueError("--dev-limit must be between 1 and 500")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    return ExperimentConfig(
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        fit_size=args.fit_size,
        threshold_size=args.threshold_size,
        dev_limit=args.dev_limit,
        batch_size=args.batch_size,
        seed=args.seed,
    )


def load_dev_sample(path: Path) -> list[PairRecord]:
    """Load the fixed 500-pair dev sample stored in the sample manifest."""
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["sample"]
    records = [
        PairRecord(
            split=row["split"],
            line_number=int(row["line_number"]),
            s1=row["s1"],
            s2=row["s2"],
            label=int(row["label"]),
        )
        for row in rows
    ]
    if len(records) != 500:
        raise ValueError(f"expected 500 dev records, got {len(records)}")
    return records


def build_clean_train_pool(
    train_path: Path,
) -> list[PairRecord]:
    """Remove train label conflicts and keep one row per canonical pair."""
    records = load_split(train_path, "train")
    groups = group_by_canonical(records)

    keep=[]
    for pairrecord_list in groups.values():
        labels={r.label for r in pairrecord_list}
        if len(labels)==1:
            first_record=min(pairrecord_list,key=lambda r:r.line_number)
            keep.append(first_record)
    keep.sort(key=lambda k:k.pair_id)
    return keep


def build_fit_and_threshold_pools(
    clean_train: list[PairRecord],
    fit_size: int,
    threshold_size: int,
    seed: int,
) -> tuple[list[PairRecord], list[PairRecord]]:
    """Select two disjoint train subsets using one fixed random seed."""
    required = fit_size + threshold_size
    if required > len(clean_train):
        raise ValueError(
            f"need {required} train records, only {len(clean_train)} available"
        )

    rng=random.Random(seed)
    sample=rng.sample(clean_train,required)
    fit_pool=sorted(sample[:fit_size], key=lambda s:s.pair_id)
    threshold_pool=sorted(sample[fit_size:],key=lambda s:s.pair_id)
    return(fit_pool,threshold_pool)
    


def assert_disjoint(
    fit_pool: list[PairRecord],
    threshold_pool: list[PairRecord],
    dev_pool: list[PairRecord],
) -> None:
    """Raise if any canonical pair crosses the three experiment pools."""
    f_canonical_key={p.canonical_key for p in fit_pool}
    t_canonical_key={p.canonical_key for p in threshold_pool}
    d_canonical_key={p.canonical_key for p in dev_pool}
    if len(f_canonical_key)!=len(fit_pool):
        raise ValueError(f"训练池内部有重复")
    if len(t_canonical_key)!=len(threshold_pool):
        raise ValueError(f"调参池有重复")
    if len(d_canonical_key)!=len(dev_pool):
        raise ValueError(f"测试池内部有重复")
    ft_both=f_canonical_key & t_canonical_key
    td_both=t_canonical_key & d_canonical_key
    fd_both=f_canonical_key & d_canonical_key
    if ft_both:
        raise ValueError("数据泄漏！fit_pool 和 threshold_pool 有交集")
    if td_both:
        raise ValueError("数据泄漏！threshold_pool 和 dev_pool 有交集")
    if fd_both:
        raise ValueError("数据泄漏！fit_pool 和 dev_pool 有交集")


def collect_unique_texts(
    records: Iterable[PairRecord],
) -> tuple[list[str], dict[str, int]]:
    """Deduplicate sentences in insertion order and build text-to-index."""
    all_text=[]
    for record in records:
        all_text.append(record.s1)
        all_text.append(record.s2)
    unique_texts=list(dict.fromkeys(all_text))
    text_index={
        text:index
        for index,text in enumerate(unique_texts)
    }
    return unique_texts,text_index
            


def load_qwen_model(model_dir: Path) -> SentenceTransformer:
    """Load the local Qwen sentence-embedding model on CPU."""


    model=SentenceTransformer(
        str(model_dir),
        device="cpu",
        model_kwargs={"dtype":torch.float32},
    )
    return model


def encode_unique_texts(
    model: SentenceTransformer,
    unique_texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode texts into a finite, L2-normalized float32 matrix."""
    embeddings=model.encode(
        unique_texts,
        batch_size=batch_size,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    embeddings=np.asarray(embeddings,dtype=np.float32)

    expected_lines=len(unique_texts)
    if embeddings.ndim!=2 or embeddings.shape[0]!=expected_lines:
        raise ValueError(
            f"embedding 形状错误：got={embeddings.shape}, "
            f"expected first dimension={expected_lines}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding 中存在 nan 或 inf")

    norms=np.linalg.norm(embeddings,axis=1,keepdims=True)
    if np.any(norms<=0):
        raise ValueError("存在零范数 embedding，无法归一化")
    embeddings=embeddings/norms
    normalized_norms=np.linalg.norm(embeddings,axis=1)
    max_norm_error=float(np.max(np.abs(normalized_norms-1.0)))
    if max_norm_error>1e-5:
        raise ValueError(
            f"embedding 归一化误差过大：{max_norm_error}"
        )
    return embeddings

def compute_pair_cosines(
    records: list[PairRecord],
    embeddings: np.ndarray,
    text_index: dict[str, int],
) -> list[ScoredPair]:
    """Compute one cosine score for every record."""
    scoredpairs=[]
    for record in records:
        s1_index=text_index[record.s1]
        s2_index=text_index[record.s2]
        score=float(np.dot(embeddings[s1_index],embeddings[s2_index]))
        scoredpairs.append(ScoredPair(record=record,score=score))
    return scoredpairs


def predict_at_threshold(
    scores: list[float],
    threshold: float,
) -> list[int]:
    """Return 1 when score >= threshold, otherwise 0."""
    predict_labellist=[]
    for score in scores:
        if score>=threshold:
            predict_labellist.append(1)
        else:
            predict_labellist.append(0)
    return predict_labellist        


def compute_metrics(
    labels: list[int],
    predictions: list[int],
) -> Metrics:
    """Compute accuracy, macro-F1 and confusion counts."""
    if not labels:
        raise ValueError("labels 不能为空")
    if len(labels)!=len(predictions):
        raise ValueError(f"预测标签数：{len(labels)}和实际标签数:{len(predictions)}不一致")
    if not set(labels) <= {0,1}:
        raise ValueError(f"(实际标签labels中存在错误标签)")
    if not set(predictions) <= {0,1}:
        raise ValueError("预测标签predictions中存在错误标签")
    TP,TN,FP,FN=0,0,0,0
    for true_label,pre_label in zip(labels,predictions):
        if true_label==1 and pre_label==1:
            TP+=1
        elif true_label==1 and pre_label==0:
            FN+=1
        elif true_label==0 and pre_label==0:
            TN+=1
        elif true_label==0 and pre_label==1:
            FP+=1
    precision_1=TP/(TP+FP) if TP+FP>0 else 0.0
    recall_1=TP/(TP+FN) if TP+FN>0 else 0.0
    precision_0=TN/(TN+FN) if TN+FN>0 else 0.0
    recall_0=TN/(TN+FP) if TN+FP>0 else 0.0
    F1_1=(
        2*precision_1*recall_1/(precision_1+recall_1)
        if precision_1+recall_1>0
        else 0.0
    )
    F1_0=(
        2*precision_0*recall_0/(precision_0+recall_0)
        if precision_0+recall_0>0
        else 0.0
    )
    macro_F1=(F1_1+F1_0)/2.0
    return Metrics(
        accuracy=(TP+TN)/len(labels),
        macro_f1=macro_F1,
        true_positive=TP,
        true_negative=TN,
        false_negative=FN,
        false_positive=FP
    )


def compute_roc_auc(
    labels: list[int],
    scores: list[float],
) -> float:
    """Compute threshold-independent ROC-AUC."""
    if not labels:
        raise ValueError("labels 不能为空")
    if len(labels) != len(scores):
        raise ValueError("labels 与 scores 长度不一致")
    if not set(labels) <= {0, 1}:
        raise ValueError("labels 只能包含 0 和 1")
    if len(set(labels)) < 2:
        raise ValueError("ROC-AUC 需要同时包含正类和负类")
    return float(roc_auc_score(labels, scores))


def select_threshold(
    scored_pairs: list[ScoredPair],
) -> ThresholdSelection:
    """Choose the threshold with highest macro-F1 on train threshold data."""
    if not scored_pairs:
        raise ValueError("threshold pool 不能为空")

    scores=[]
    labels=[]
    best_threshold=0.0
    best_macro_f1=-1.0
    best_metrics:Metrics|None=None
    for scored_pair in scored_pairs:
        scores.append(scored_pair.score)
        labels.append(scored_pair.record.label)

    min_score=min(scores)
    max_score=max(scores)
    thresholds=sorted(
        set(scores)
        | {min_score-1e-12,max_score+1e-12}
    )
    for threshold in thresholds:
        predictions=predict_at_threshold(scores,threshold)
        metrics=compute_metrics(labels,predictions)
        if (
            metrics.macro_f1>best_macro_f1
            or (
                metrics.macro_f1==best_macro_f1
                and threshold>best_threshold
            )
        ):
            best_macro_f1=metrics.macro_f1
            best_threshold=threshold
            best_metrics=metrics
    
    if best_metrics is None:
        raise RuntimeError("没有找到有效阈值")
    return ThresholdSelection(
        threshold=best_threshold,
        threshold_metrics=best_metrics
    )

        

def make_pair_text(record: PairRecord) -> str:
    """Build one text feature string for TF-IDF + LogisticRegression."""
    return f"{record.s1} [SEP] {record.s2}"


def train_tfidf_baseline(
    fit_pool: list[PairRecord],
) -> tuple[Any, Any]:
    """Fit the fixed TF-IDF vectorizer and logistic regression model."""
    fit_pool_texts=[
        make_pair_text(pairrecord)
        for pairrecord in fit_pool
    ]
    fit_pool_labels=[
        pairrecord.label
        for pairrecord in fit_pool
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=2,
    )
    features_fit_pool=vectorizer.fit_transform(fit_pool_texts)
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(features_fit_pool,fit_pool_labels)
    return vectorizer,model

def predict_tfidf(
    vectorizer: Any,
    classifier: Any,
    records: list[PairRecord],
) -> tuple[list[int], list[float]]:
    """Predict labels and positive probabilities without refitting TF-IDF."""
    texts=[]
    for record in records:
        texts.append(make_pair_text(record))
    records_features=vectorizer.transform(texts)
    predictions=classifier.predict(records_features)
    probabilities = classifier.predict_proba(records_features)[:, 1]
    return (
        [int(prediction) for prediction in predictions],
        [float(probability) for probability in probabilities],
    )
    


def write_scored_pairs(
    path: Path,
    scored_pairs: list[ScoredPair],
    predictions: list[int],
    score_column: str,
) -> None:
    """Write traceable per-pair prediction data as CSV."""
    if len(scored_pairs) != len(predictions):
        raise ValueError(
            f"scored_pairs={len(scored_pairs)} but "
            f"predictions={len(predictions)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames=["pair_id","sentence_a","sentence_b","label","prediction",score_column]
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        for sp,pre in zip(scored_pairs,predictions):
            row={
             "pair_id":sp.record.pair_id,
             "sentence_a":sp.record.s1,
             "sentence_b":sp.record.s2,
             "label":sp.record.label,
             "prediction":pre,
             score_column:sp.score
            }
            writer.writerow(row)



def collect_errors(
    records: list[PairRecord],
    predictions: list[int],
    scores: list[float],
) -> list[dict[str, object]]:
    """Return FP and FN rows with original text and label."""
    if not (len(records) == len(predictions) == len(scores)):
        raise ValueError("records、predictions、scores 长度不一致")
    error_list=[]
    for r, pre, score in zip(records,predictions,scores):
        if r.label==pre:
            continue
        if r.label==0 and pre==1:
            error_type="false_positive"
        elif r.label==1 and pre==0:
            error_type="false_negative"
        else:
            raise ValueError("预测值必须是 0 或 1")
        error_list.append({
            "pair_id":r.pair_id,
            "split":r.split,
            "line_number":r.line_number,
            "sentence_1":r.s1,
            "sentence_2":r.s2,
            "prediction":int(pre),
            "label":int(r.label),
            "score":float(score),
            "error_type":error_type,
        })
    return error_list


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w",encoding="utf-8")as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2, 
            sort_keys=True
        )


def write_experiment_report(
    path: Path,
    config: ExperimentConfig,
    qwen_threshold: ThresholdSelection,
    qwen_metrics: Metrics,
    tfidf_metrics: Metrics,
    qwen_auc: float,
    tfidf_auc: float,
) -> None:
    """Write a concise Markdown report with fixed comparison caveats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    train_path = config.data_dir / "train.txt"
    report = (
        "# Qwen vs TF-IDF LCQMC 实验\n"
        "\n"
        "## 实验口径\n"
        "\n"
        "- 本实验比较两条端到端文本相似度流程。\n"
        "- TF-IDF + 逻辑回归是有监督分类器。\n"
        "- Qwen 句向量 + 余弦阈值是无监督相似度加人工阈值。\n"
        "- 因此结果不能单独归因于表示模型本身。\n"
        "- Qwen 阈值只在 train threshold 集选择；dev 只用于最终评估。\n"
        "- test 未使用。\n"
        "\n"
        "## 数据规模\n"
        f"\n"
        f"- TF-IDF fit records: {config.fit_size}\n"
        f"- Qwen threshold records: {config.threshold_size}\n"
        f"- Dev records: {config.dev_limit}\n"
        f"- Random seed: {config.seed}\n"
        "\n"
        "## 数据来源校验\n"
        "\n"
        f"- train.txt SHA-256: `{sha256_file(train_path)}`\n"
        f"- sample manifest SHA-256: `{sha256_file(config.manifest_path)}`\n"
        "\n"
        "## 结果\n"
        "\n"
        "| Flow | Accuracy | Macro-F1 | ROC-AUC | TP | TN | FP | FN |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| Qwen + threshold | {qwen_metrics.accuracy:.6f} | "
        f"{qwen_metrics.macro_f1:.6f} | {qwen_auc:.6f} | "
        f"{qwen_metrics.true_positive} | "
        f"{qwen_metrics.true_negative} | {qwen_metrics.false_positive} | "
        f"{qwen_metrics.false_negative} |\n"
        f"| TF-IDF + LR | {tfidf_metrics.accuracy:.6f} | "
        f"{tfidf_metrics.macro_f1:.6f} | {tfidf_auc:.6f} | "
        f"{tfidf_metrics.true_positive} | "
        f"{tfidf_metrics.true_negative} | {tfidf_metrics.false_positive} | "
        f"{tfidf_metrics.false_negative} |\n"
        "\n"
        "## Qwen 阈值\n"
        "\n"
        f"- Selected threshold: {qwen_threshold.threshold:.12g}\n"
        f"- Threshold-set accuracy: "
        f"{qwen_threshold.threshold_metrics.accuracy:.6f}\n"
        f"- Threshold-set macro-F1: "
        f"{qwen_threshold.threshold_metrics.macro_f1:.6f}\n"
        "\n"
        "## 限制\n"
        "\n"
        "- 当前 dev 样本只属于固定小样本试点，不能替代完整 dev 或泛化结论。\n"
        "- 两条流程的分类机制不同，指标差异不能只归因于 embedding。\n"
        "- 当前没有训练或微调 Qwen。\n"
        "- 错例标签和类别仍需人工复核。\n"
    )
    path.write_text(report, encoding="utf-8", newline="\n")
    


def main() -> int:
    config = make_config(parse_args())

    train_path = config.data_dir / "train.txt"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    if not config.manifest_path.is_file():
        raise FileNotFoundError(config.manifest_path)
    if not config.model_dir.is_dir():
        raise FileNotFoundError(config.model_dir)

    clean_train = build_clean_train_pool(train_path)
    fit_pool, threshold_pool = build_fit_and_threshold_pools(
        clean_train=clean_train,
        fit_size=config.fit_size,
        threshold_size=config.threshold_size,
        seed=config.seed,
    )
    dev_pool = load_dev_sample(config.manifest_path)[: config.dev_limit]
    assert_disjoint(fit_pool, threshold_pool, dev_pool)

    all_qwen_records = threshold_pool + dev_pool

    unique_texts, text_index = collect_unique_texts(all_qwen_records)

    model = load_qwen_model(config.model_dir)
    embeddings = encode_unique_texts(
        model=model,
        unique_texts=unique_texts,
        batch_size=config.batch_size,
    )

    threshold_scores = compute_pair_cosines(
        threshold_pool, embeddings, text_index
    )
    threshold_selection = select_threshold(threshold_scores)
    dev_scores = compute_pair_cosines(dev_pool, embeddings, text_index)

    qwen_predictions = predict_at_threshold(
        [item.score for item in dev_scores],
        threshold_selection.threshold,
    )
    qwen_metrics = compute_metrics(
        [item.record.label for item in dev_scores],
        qwen_predictions,
    )
    qwen_auc = compute_roc_auc(
        [item.record.label for item in dev_scores],
        [item.score for item in dev_scores],
    )

    vectorizer, classifier = train_tfidf_baseline(fit_pool)
    tfidf_predictions, tfidf_probabilities = predict_tfidf(
        vectorizer, classifier, dev_pool
    )
    tfidf_metrics = compute_metrics(
        [record.label for record in dev_pool],
        tfidf_predictions,
    )
    tfidf_auc = compute_roc_auc(
        [record.label for record in dev_pool],
        tfidf_probabilities,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_scored_pairs(
        config.output_dir / "qwen_dev_scores.csv",
        dev_scores,
        qwen_predictions,
        score_column="cosine",
    )
    write_scored_pairs(
        config.output_dir / "tfidf_dev_scores.csv",
        [
            ScoredPair(record=record, score=probability)
            for record, probability in zip(dev_pool, tfidf_probabilities)
        ],
        tfidf_predictions,
        score_column="positive_probability",
    )
    with (config.output_dir / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "accuracy",
                "macro_f1",
                "roc_auc",
                "true_positive",
                "true_negative",
                "false_positive",
                "false_negative",
            ],
        )
        writer.writeheader()
        for model_name, metrics, roc_auc in (
            ("qwen_threshold", qwen_metrics, qwen_auc),
            ("tfidf_lr", tfidf_metrics, tfidf_auc),
        ):
            writer.writerow(
                {
                    "model": model_name,
                    "accuracy": metrics.accuracy,
                    "macro_f1": metrics.macro_f1,
                    "roc_auc": roc_auc,
                    "true_positive": metrics.true_positive,
                    "true_negative": metrics.true_negative,
                    "false_positive": metrics.false_positive,
                    "false_negative": metrics.false_negative,
                }
            )

    write_json(
        config.output_dir / "metrics.json",
        {
            "qwen": qwen_metrics.__dict__,
            "tfidf": tfidf_metrics.__dict__,
            "qwen_auc": qwen_auc,
            "tfidf_auc": tfidf_auc,
            "qwen_threshold": threshold_selection.threshold,
            "qwen_threshold_macro_f1":
                threshold_selection.threshold_metrics.macro_f1,
            "train_hash": sha256_file(train_path),
            "dev_manifest_hash": sha256_file(config.manifest_path),
        },
    )
    write_json(
        config.output_dir / "errors_qwen.json",
        {
            "errors": collect_errors(
                dev_pool,
                qwen_predictions,
                [item.score for item in dev_scores],
            )
        },
    )
    write_json(
        config.output_dir / "errors_tfidf.json",
        {
            "errors": collect_errors(
                dev_pool,
                tfidf_predictions,
                tfidf_probabilities,
            )
        },
    )
    write_experiment_report(
        config.output_dir / "README.md",
        config,
        threshold_selection,
        qwen_metrics,
        tfidf_metrics,
        qwen_auc,
        tfidf_auc,
    )

    print("fit records:", len(fit_pool))
    print("threshold records:", len(threshold_pool))
    print("dev records:", len(dev_pool))
    print("unique texts:", len(unique_texts))
    print("threshold:", threshold_selection.threshold)
    print("qwen metrics:", qwen_metrics)
    print("tfidf metrics:", tfidf_metrics)
    print("qwen auc:", qwen_auc)
    print("tfidf auc:", tfidf_auc)
    print("output:", config.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
