"""Qwen 冻结分类头小试点的工程外围，不代写学生特征或拟合核心。"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import inspect
from importlib.metadata import version
import json
import random
import sys
import textwrap
import time
import warnings

ROOT = Path(__file__).resolve().parent
JOURNEY = ROOT.parent
sys.path.insert(0, str(JOURNEY / "w8-d1"))
from sample_pairs import load_split, group_by_canonical

DATA = Path("D:/Desktop/校/计算机/datasets/LCQMC")
PILOT = JOURNEY / "w8-d1/artifacts/sample_manifest.json"
MODEL = JOURNEY / "models/Qwen3-Embedding-0.6B"
MANIFEST = ROOT / "data/smoke_manifest.json"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonical(row):
    return tuple(sorted((row["s1"], row["s2"])))


def serialize(record):
    return {"pair_id": record.pair_id, "split": record.split, "line_number": record.line_number,
            "s1": record.s1, "s2": record.s2, "label": record.label}


def require_same_existing(path, obj):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != obj:
            raise ValueError(f"已有清单与当前选择不符，不自动覆盖：{path}")
    else:
        dump(path, obj)


def prepare_manifest(path=MANIFEST):
    # 与 W8 的阈值池选择保持一致，不导入会加载深度学习库的旧实验脚本。
    groups = group_by_canonical(load_split(DATA / "train.txt", "train"))
    clean = [min(rows, key=lambda r: r.line_number) for rows in groups.values()
             if len({r.label for r in rows}) == 1]
    clean.sort(key=lambda r: r.pair_id)
    sampled = random.Random(42).sample(clean, 22000)
    threshold = sorted(sampled[20000:], key=lambda r: r.pair_id)
    by_label = {label: [r for r in threshold if r.label == label][:10] for label in (0, 1)}
    if any(len(rows) != 10 for rows in by_label.values()):
        raise ValueError("原阈值池无法提供两类各十对")
    train = [serialize(by_label[label][i]) for i in range(10) for label in (1, 0)]
    pilot_document = json.loads(PILOT.read_text(encoding="utf-8"))
    if pilot_document.get("data_hashes") != {"train.txt": sha(DATA / "train.txt"),
                                            "dev.txt": sha(DATA / "dev.txt")}:
        raise ValueError("原 dev500 清单记录的数据源指纹不符")
    pilot = pilot_document["sample"]
    if len(pilot) != 500:
        raise ValueError("原 dev 试点必须恰好500对")
    dev = [{**row, "pair_id": f"dev:{int(row['line_number']):06d}"} for row in pilot[:20]]
    verify_records(train, dev)
    obj = {"schema": 1, "scope": "smoke_only_train20_dev20_from_used_fixed_dev500",
           "selection": {"train": "10_per_label_from_W8_threshold2000_fit20000_seed42",
                         "dev": "first20_in_W8_fixed_dev500_manifest"},
           "source_hashes": {"train.txt": sha(DATA / "train.txt"), "dev.txt": sha(DATA / "dev.txt"),
                             "fixed_dev500_manifest": sha(PILOT)},
           "threshold_pool_ids_sha256": hashlib.sha256(
               "\n".join(r.pair_id for r in threshold).encode()).hexdigest(),
           "train": train, "dev_diagnostic": dev}
    require_same_existing(path, obj)
    return obj


def verify_records(train, dev):
    for split, rows in (("train", train), ("dev", dev)):
        if not rows or any(r["split"] != split or r["label"] not in (0, 1) for r in rows):
            raise ValueError("样本划分或标签不符")
        if len({canonical(r) for r in rows}) != len(rows):
            raise ValueError("样本内部存在相同/交换句序重复")
        expected = {r["line_number"]: r for r in rows}
        if len(expected) != len(rows):
            raise ValueError("样本原始行号重复")
        found = set()
        with (DATA / f"{split}.txt").open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if number in expected:
                    a, b, label = line.rstrip("\r\n").split("\t")
                    row = expected[number]
                    if (a, b, int(label), f"{split}:{number:06d}") != (
                            row["s1"], row["s2"], row["label"], row["pair_id"]):
                        raise ValueError("句对、标签、ID 与原始行不对应")
                    found.add(number)
                if len(found) == len(rows):
                    break
        if len(found) != len(rows):
            raise ValueError("样本行号超出数据源")
    if {canonical(r) for r in train} & {canonical(r) for r in dev}:
        raise ValueError("train 与 dev 句对重叠")
    if {r["label"] for r in train} != {0, 1}:
        raise ValueError("训练样本需要两类")


def read_manifest(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    # 既验证原始行，也验证“来自原阈值池的既定20条”这一选择声明。
    prepare_manifest(path)
    hashes = obj["source_hashes"]
    if hashes != {"train.txt": sha(DATA / "train.txt"), "dev.txt": sha(DATA / "dev.txt"),
                  "fixed_dev500_manifest": sha(PILOT)}:
        raise ValueError("数据源或原固定500清单已变化")
    train, dev = obj["train"], obj["dev_diagnostic"]
    if len(train) != 20 or len(dev) != 20:
        raise ValueError("本日实验只接受20对train、20对dev诊断子集")
    verify_records(train, dev)
    original = json.loads(PILOT.read_text(encoding="utf-8"))["sample"][:20]
    if [r["line_number"] for r in dev] != [r["line_number"] for r in original]:
        raise ValueError("dev必须来自原固定500的前20对")
    return obj


def text_layout(obj):
    texts = list(dict.fromkeys(text for row in obj["train"] + obj["dev_diagnostic"]
                              for text in (row["s1"], row["s2"])))
    return texts, {text: index for index, text in enumerate(texts)}


def encoding_contract(model_dir):
    paths = [p for p in model_dir.rglob("*") if p.is_file() and
             (p.suffix in (".json", ".safetensors", ".txt") or p.name == "merges.txt")]
    files = {str(p.relative_to(model_dir)).replace("\\", "/"): sha(p) for p in sorted(paths)}
    if "model.safetensors" not in files:
        raise ValueError("未找到Qwen权重")
    return {"model_files": files, "dtype": "float32", "device": "cpu", "prompt": None,
            "normalize": "L2_float32_after_encode", "pooling": "local_sentence_transformer_modules",
            "batch_size": 2, "max_seq_length": 512, "threads": 4}


def check_embeddings(embeddings, count):
    import numpy as np
    if embeddings.shape != (count, 1024) or embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError("缓存必须是有限 float32 [unique_texts,1024]")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5):
        raise ValueError("缓存向量必须L2归一化")


def check_cache_versions(metadata):
    actual = {"torch": version("torch"), "numpy": version("numpy"),
              "sentence_transformers": version("sentence-transformers")}
    if metadata["versions"] != actual:
        raise ValueError("当前库版本与缓存生成版本不一致，请先核对，不能静默沿用")
    if not metadata.get("all_requires_grad_false") or not metadata.get("all_grad_none"):
        raise ValueError("缓存缺少冻结编码的检查记录")
    fingerprints = [metadata.get(name) for name in ("frozen_weights_before", "frozen_weights_after")]
    if any(not isinstance(value, str) or len(value) != 64 or
           any(character not in "0123456789abcdefABCDEF" for character in value)
           for value in fingerprints):
        raise ValueError("编码前后权重指纹必须是非空的64位十六进制SHA256字符串")
    if fingerprints[0] != fingerprints[1]:
        raise ValueError("编码前后权重指纹不一致")


def fingerprint_parameters(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        digest.update(name.encode())
        # CPU float32 的视图，不复制整个大矩阵。
        array = param.detach().cpu().contiguous().numpy()
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def encode_cache(manifest, manifest_path, model_dir, output):
    import numpy as np
    obj = read_manifest(manifest_path)
    texts, _ = text_layout(obj)
    contract = encoding_contract(model_dir)
    target = output / "embeddings.npy"
    metadata_path = output / "cache.json"
    identity = {"manifest_sha256": sha(manifest_path), "texts": texts, "encoding": contract}
    if target.exists() and metadata_path.exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        check_cache_versions(cached)
        if cached["identity"] != identity or sha(target) != cached["embeddings_sha256"]:
            raise ValueError("已有缓存身份不符；先定位差异，不静默重算覆盖")
        check_embeddings(np.load(target, allow_pickle=False), len(texts))
        print(f"CACHE_READY reused=True unique_texts={len(texts)}", flush=True)
        return
    import torch
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    torch.manual_seed(42)
    print(f"Loading frozen Qwen; unique_texts={len(texts)}", flush=True)
    start = time.perf_counter()
    model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True,
                                model_kwargs={"dtype": torch.float32})
    model.max_seq_length = 512
    model.default_prompt_name = None
    model.requires_grad_(False)
    model.eval()
    load_seconds = time.perf_counter() - start
    lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
    if max(lengths) > model.max_seq_length:
        raise ValueError("出现超长句子，不默默截断，请核对编码口径")
    before = fingerprint_parameters(model)
    start = time.perf_counter()
    with torch.inference_mode():
        vectors = model.encode(texts, batch_size=2, normalize_embeddings=False, show_progress_bar=True,
                               convert_to_numpy=True, prompt_name=None)
    encode_seconds = time.perf_counter() - start
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("发现零范数向量")
    vectors = np.asarray(vectors / norms, dtype=np.float32)
    check_embeddings(vectors, len(texts))
    after = fingerprint_parameters(model)
    if before != after or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError("Qwen 冻结验证未通过")
    output.mkdir(parents=True, exist_ok=True)
    np.save(target, vectors, allow_pickle=False)
    dump(metadata_path, {"identity": identity, "embeddings_sha256": sha(target),
                         "versions": {"torch": torch.__version__, "numpy": np.__version__,
                                      "sentence_transformers": sentence_transformers.__version__},
                         "frozen_weights_before": before, "frozen_weights_after": after,
                         "all_requires_grad_false": True, "all_grad_none": True,
                         "load_seconds": load_seconds, "encode_seconds": encode_seconds,
                         "max_token_length": max(lengths), "shape": list(vectors.shape)})
    print(f"CACHE_READY reused=False shape={vectors.shape} encode_seconds={encode_seconds:.3f}", flush=True)


def load_pair_arrays(obj, manifest_path, model_dir, output):
    import numpy as np
    meta = json.loads((output / "cache.json").read_text(encoding="utf-8"))
    check_cache_versions(meta)
    texts, index = text_layout(obj)
    identity = {"manifest_sha256": sha(manifest_path), "texts": texts, "encoding": encoding_contract(model_dir)}
    path = output / "embeddings.npy"
    if meta["identity"] != identity or meta["embeddings_sha256"] != sha(path):
        raise ValueError("缓存与句对/编码设置不匹配")
    vectors = np.load(path, allow_pickle=False)
    check_embeddings(vectors, len(texts))
    result = []
    for rows in (obj["train"], obj["dev_diagnostic"]):
        left = np.stack([vectors[index[r["s1"]]] for r in rows])
        right = np.stack([vectors[index[r["s2"]]] for r in rows])
        labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
        result.append((left, right, labels))
    return result


def is_pending(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    for statement in tree.body[0].body:
        if isinstance(statement, ast.Return):
            return False
        if isinstance(statement, ast.Raise):
            call = statement.exc.func if isinstance(statement.exc, ast.Call) else statement.exc
            if isinstance(call, ast.Name) and call.id == "NotImplementedError":
                return True
    return False


def validate_features(function, left, right):
    import numpy as np
    if left.ndim != 2 or left.shape != right.shape or min(left.shape) <= 0:
        raise ValueError("左右输入必须是形状相同的非空二维矩阵")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("输入有非有限数")
    # 独立固定答案可排除“形状和对称性都正确，但全部返回零”的错误。
    toy_left = np.array([[1, -2], [0, 3]], dtype=np.float32)
    toy_right = np.array([[-1, 2], [4, 1]], dtype=np.float32)
    expected = np.array([[2, 4, -1, -4], [4, 2, 0, 3]], dtype=np.float32)
    toy = function(toy_left, toy_right)
    if not isinstance(toy, np.ndarray) or toy.shape != expected.shape or not np.allclose(toy, expected):
        raise AssertionError("固定小例子不符，请核对绝对差、逐维乘积和横向拼接次序")
    a, b = left.copy(), right.copy()
    features = function(left, right)
    if not isinstance(features, np.ndarray) or features.shape != (len(left), 2 * left.shape[1]):
        raise AssertionError("句对特征形状必须为 [N,2D]，检查是否沿列拼接")
    if features.dtype != np.float32 or not np.isfinite(features).all():
        raise AssertionError("特征必须是有限 float32")
    if not np.array_equal(left, a) or not np.array_equal(right, b):
        raise AssertionError("不应修改原始句向量")
    swapped = function(right.copy(), left.copy())
    if not np.allclose(features, swapped, rtol=1e-6, atol=1e-7):
        raise AssertionError("交换句序后特征应不变")
    return features


def fit_and_predict(obj, args):
    import joblib
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.utils.validation import check_is_fitted
    from student_head import make_pair_features, fit_head
    if is_pending(make_pair_features) or is_pending(fit_head):
        print("STUDENT_TODO: 请完成 student_head.py 的两个函数；尚未训练分类头。")
        raise SystemExit(2)
    arrays = load_pair_arrays(obj, args.manifest, args.model_dir, args.output_dir)
    left, right, labels = arrays[0]
    dev_left, dev_right, dev_labels = arrays[1]
    train_features = validate_features(make_pair_features, left, right)
    dev_features = validate_features(make_pair_features, dev_left, dev_right)
    classifier = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    cache_hash_before = sha(args.output_dir / "embeddings.npy")
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        returned = fit_head(classifier, train_features.copy(), labels.copy())
    seconds = time.perf_counter() - start
    if returned is not classifier:
        raise AssertionError("fit_head 应返回同一个已拟合分类头")
    check_is_fitted(classifier)
    if classifier.n_features_in_ != 2048 or classifier.classes_.tolist() != [0, 1]:
        raise AssertionError("分类头维度或标签类别不符")
    if not np.isfinite(classifier.coef_).all() or not np.any(classifier.coef_ != 0):
        raise AssertionError("未取得有效的分类头系数")
    predictions = classifier.predict(dev_features)
    probabilities = classifier.predict_proba(dev_features)[:, 1]
    if sha(args.output_dir / "embeddings.npy") != cache_hash_before:
        raise AssertionError("拟合分类头时不应修改句向量缓存")
    before_swapped = classifier.predict_proba(validate_features(make_pair_features, dev_right, dev_left))
    if not np.allclose(before_swapped[:, 1], probabilities, atol=1e-7):
        raise AssertionError("交换句序后预测应一致")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dev20_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "s1", "s2", "label", "prediction", "positive_probability"])
        writer.writeheader()
        for row, pred, prob in zip(obj["dev_diagnostic"], predictions, probabilities):
            writer.writerow({k: row[k] for k in ("pair_id", "s1", "s2", "label")} |
                            {"prediction": int(pred), "positive_probability": float(prob)})
    joblib.dump(classifier, args.output_dir / "head.joblib")
    reloaded = joblib.load(args.output_dir / "head.joblib")
    if not np.allclose(reloaded.predict_proba(dev_features)[:, 1], probabilities, atol=1e-12):
        raise AssertionError("分类头保存后重载的预测不一致")
    report = {"status": "HEAD_SMOKE_OK", "scope": obj["scope"], "train_pairs": 20,
              "dev_diagnostic_pairs": 20, "feature_shape": list(train_features.shape),
              "manifest_sha256": sha(args.manifest), "cache_sha256": cache_hash_before,
              "student_source_sha256": sha(ROOT / "student_head.py"), "fit_seconds": seconds,
              "head": classifier.get_params(), "coef_norm": float(np.linalg.norm(classifier.coef_)),
              "dev_diagnostic_accuracy": float(accuracy_score(dev_labels, predictions)),
              "dev_diagnostic_macro_f1": float(f1_score(dev_labels, predictions, average="macro", zero_division=0)),
              "confusion_matrix_labels_0_1": confusion_matrix(dev_labels, predictions, labels=[0, 1]).tolist(),
              "cache_unchanged": True, "swapped_prediction_equal": True,
              "reload_prediction_equal": True, "interpretation_by_student": "待本人解释；小样本试通不作性能结论"}
    dump(args.output_dir / "head_report.json", report)
    print(f"HEAD_SMOKE_OK train=20 dev_diagnostic=20 features={train_features.shape} fit_seconds={seconds:.4f}")
    print("仅是已使用dev500中的20对诊断子集；不是完整dev，也不是独立test。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "encode", "inspect", "fit"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if args.stage == "prepare":
        obj = prepare_manifest(args.manifest)
        texts, _ = text_layout(obj)
        print(f"SAMPLES_READY train=20 dev_diagnostic=20 unique_texts={len(texts)}")
        return
    obj = read_manifest(args.manifest)
    if args.stage == "encode":
        encode_cache(obj, args.manifest, args.model_dir, args.output_dir)
    elif args.stage == "inspect":
        train, dev = load_pair_arrays(obj, args.manifest, args.model_dir, args.output_dir)
        print(f"CACHE_INSPECT_OK train_left={train[0].shape} dev_left={dev[0].shape} train_labels={train[2].shape}")
    else:
        fit_and_predict(obj, args)


if __name__ == "__main__":
    main()
