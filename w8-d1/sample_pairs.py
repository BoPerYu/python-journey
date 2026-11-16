from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import random
import hashlib
import json

DEFAULT_DATA_DIR = Path(r"D:\Desktop\校\计算机\datasets\LCQMC")

@dataclass(frozen=True)
class PairRecord:
    split: str
    line_number: int
    s1: str
    s2: str
    label: int

    @property
    def pair_id(self) -> str:
        return f"{self.split}:{self.line_number:06d}"

    @property
    def canonical_key(self) -> tuple[str, str]:
        a,b=sorted((self.s1,self.s2))
        return (a,b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit LCQMC train/dev overlap and build a clean dev sample."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing train.txt and dev.txt.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of clean dev records to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/sample_manifest.json"),
        help="Output manifest path.",
    )
    return parser.parse_args()


def load_split(path: Path, split: str) -> list[PairRecord]:
    # TODO:
    # 1. 严格按 UTF-8 读取 path。
    # 2. every line 恰好拆成三列。
    # 3. label 必须能转成 0 或 1。
    # 4. 按行号构造 PairRecord。
    # 5. 坏行报错时必须包含 split 和 line_number。
    with path.open("r",encoding="utf-8") as f:
        records=[]
        for i,line in enumerate(f,start=1):
            line=line.rstrip('\r\n')
            fields=line.split("\t")
            s1,s2,label_text=fields
            if len(fields)!=3:
                raise ValueError(f"第{i}行，字段数不是3")
            label=int(label_text)
            if label not in (0,1):
                raise ValueError(f"第{i}行，标签错误")
            record=PairRecord(
                split=split,
                line_number=i,
                s1=s1,
                s2=s2,
                label=label,
            )
            records.append(record)
        return records


def group_by_canonical(
    records: list[PairRecord],
) -> dict[tuple[str, str], list[PairRecord]]:
    # TODO:
    # 1. 创建 defaultdict(list)。
    # 2. 按 record.canonical_key 分组。
    # 3. 返回普通 dict 或 defaultdict 均可，但类型约定要一致。
    groups=defaultdict(list)
    for record in records:
        key=record.canonical_key
        groups[key].append(record)
    return groups


def audit_split(records: list[PairRecord]) -> dict[str, object]:
    # TODO:
    # 至少统计：
    # records / canonical_pairs / duplicate_groups /
    # duplicate_extra_rows / label_conflict_groups。
    groups=group_by_canonical(records)
    duplicate_groups=0
    duplicate_extra_rows=0
    label_conflict_groups=0
    for group in groups.values():
        if len(group)>1:
            duplicate_groups+=1
            duplicate_extra_rows+=len(group)-1
        labels={record.label for record in group}
        if len(labels)>1:
            label_conflict_groups+=1
    return {
        "records": len(records),
        "canonical_pairs": len(groups),
        "duplicate_groups": duplicate_groups,
        "duplicate_extra_rows": duplicate_extra_rows,
        "label_conflict_groups": label_conflict_groups,
    }

def audit_cross_split(
    train: list[PairRecord],
    dev: list[PairRecord],
) -> dict[str, object]:
    # TODO:
    # 按 canonical_key 检查 train/dev 的交集与标签冲突。
    train_groups = group_by_canonical(train)
    dev_groups = group_by_canonical(dev)
    shared_keys = train_groups.keys() & dev_groups.keys()
    train_dev_overlap_pairs = len(shared_keys)
    train_dev_label_conflicts = 0
    for key in shared_keys:
        train_labels = {record.label for record in train_groups[key]}
        dev_labels = {record.label for record in dev_groups[key]}
        combined_labels=train_labels|dev_labels
        if len(combined_labels)>1:
            train_dev_label_conflicts += 1
    return{
        "train_dev_overlap_pairs":train_dev_overlap_pairs,
        "train_dev_label_conflicts":train_dev_label_conflicts,
    }

def build_clean_dev(
    train: list[PairRecord],
    dev: list[PairRecord],
) -> list[PairRecord]:
    # TODO:
    # 排除 train 中出现过的 canonical pair；
    # 排除 dev 内部标签冲突的 canonical pair；
    # 同标签重复只保留第一条。
    train_groups = group_by_canonical(train)
    dev_groups = group_by_canonical(dev)
    clean_dev=[]
    same_in_train=0
    label_friction=0
    label_repetion=0
    for key,group in dev_groups.items():
        if key in train_groups.keys():
            same_in_train+=len(group)
            continue
        labels={record.label for record in group}
        if len(labels)>1:
            label_friction+=len(group)
            continue
        first=min(group,key=lambda r:r.line_number)
        clean_dev.append(first)
        label_repetion+=len(group)-1
    print(
        f"因 train/dev 重叠而排除的 dev 行数:{same_in_train},\n因标签冲突而排除的 dev 行数{label_friction},\n因同标签重复而删掉的额外行数{label_repetion}."
    )
    return clean_dev

def sample_records(
    records: list[PairRecord],
    size: int,
    seed: int,
) -> list[PairRecord]:
    # TODO:
    # 1. 检查 size 是否合法。
    # 2. random.Random(seed).sample(...)
    # 3. 按 pair_id 排序后返回。
    if size<0 or size>len(records):
        raise ValueError(f"抽取数目{size}不得为负数且不能大于记录总数{len(records)}")
    rng=random.Random(seed)
    selected=rng.sample(records,size)
    selected.sort(key=lambda r:r.pair_id)
    return selected

def sha256_file(path: Path) -> str:
    # TODO:
    # 分块读取二进制文件并计算 SHA-256，避免一次读入大文件。
    hasher=hashlib.sha256()
    chunk_size=1024*1024
    with path.open("rb") as f:
        while True:
            chunk=f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()



def write_manifest(
    path: Path,
    data_hashes: dict[str, str],
    sample: list[PairRecord],
    seed: int,
    audit: dict[str, object],
) -> None:
    # TODO:
    # 1. 创建父目录。
    # 2. 组织 JSON 可以序列化的字典。
    # 3. 严格 UTF-8 写出。
    path.parent.mkdir(parents=True, exist_ok=True)
    input = {
        "seed":seed,
        "data_hashes":dict(data_hashes),
        "sample":[
            {
                "pair_id":record.pair_id,
                "split": record.split,
                "line_number": record.line_number,
                "s1": record.s1,
                "s2": record.s2,
                "label": record.label,
            }
            for record in sample
        ],
        "audit":dict(audit),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            input, 
            f, 
            ensure_ascii=False, 
            indent=2, 
            sort_keys=True,
            default=str
            )


def main() -> int:
    args = parse_args()

    train_path = args.data_dir / "train.txt"
    dev_path = args.data_dir / "dev.txt"
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("train.txt 或 dev.txt 不存在")

    train = load_split(train_path, "train")
    dev = load_split(dev_path, "dev")

    audit = {
        "train": audit_split(train),
        "dev": audit_split(dev),
        "cross_split": audit_cross_split(train, dev),
    }

    clean_dev = build_clean_dev(train, dev)
    sample = sample_records(clean_dev, args.sample_size, args.seed)
    hashes = {
        "train.txt": sha256_file(train_path),
        "dev.txt": sha256_file(dev_path),
    }

    write_manifest(
        path=args.report,
        data_hashes=hashes,
        sample=sample,
        seed=args.seed,
        audit=audit,
    )

    print("train records:", len(train))
    print("dev records:", len(dev))
    print("clean dev records:", len(clean_dev))
    print("sample size:", len(sample))
    print("manifest:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
