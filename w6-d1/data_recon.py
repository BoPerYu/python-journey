r"""Inspect the LCQMC TSV files without loading them into pandas.

The script reports file encoding, record counts, label distribution, sentence
length statistics, parse errors, and newline style. It treats sentence length
as the number of Unicode code points returned by len(), not bytes or tokens.

Run from this directory:
    ..\w5-d1\.venv\Scripts\python.exe -X utf8 data_recon.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import statistics


SPLITS = ("train", "dev", "test")
LENGTH_PERCENTILES = (50, 90, 95, 99)


def percentile(values: list[int], pct: int) -> float:
    """Return a linearly interpolated percentile for sorted numeric values."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        **{
            f"p{pct}": round(percentile(values, pct), 3)
            for pct in LENGTH_PERCENTILES
        },
        "max": max(values),
    }


def decode_utf8(raw: bytes) -> tuple[str, str]:
    """Decode LCQMC strictly and report whether a UTF-8 BOM was present."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "UTF-8 with BOM"
    return raw.decode("utf-8"), "UTF-8 without BOM"


def inspect_split(path: Path, expects_label: bool) -> dict[str, object]:
    raw = path.read_bytes()
    text, encoding = decode_utf8(raw)
    lines = text.splitlines()

    label_counts: Counter[int] = Counter()
    s1_lengths: list[int] = []
    s2_lengths: list[int] = []
    parse_errors: list[dict[str, object]] = []
    parse_error_count = 0
    empty_field_count = 0

    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        expected_columns = 3 if expects_label else 2
        error = None

        if len(fields) != expected_columns:
            error = f"expected {expected_columns} columns, got {len(fields)}"
        elif any(field == "" for field in fields):
            empty_field_count += 1
            error = "contains an empty field"
        elif expects_label:
            try:
                label = int(fields[2])
            except ValueError:
                error = f"label is not an integer: {fields[2]!r}"
            else:
                if label not in (0, 1):
                    error = f"label is outside 0/1: {label}"

        if error is not None:
            parse_error_count += 1
            if len(parse_errors) < 5:
                parse_errors.append(
                    {"line": line_number, "error": error, "content": line}
                )
            continue

        s1, s2 = fields[0], fields[1]
        s1_lengths.append(len(s1))
        s2_lengths.append(len(s2))
        if expects_label:
            label_counts[int(fields[2])] += 1

    record_count = len(s1_lengths)
    sentence_lengths = s1_lengths + s2_lengths
    crlf_count = raw.count(b"\r\n")
    lf_count = raw.count(b"\n") - crlf_count

    result: dict[str, object] = {
        "path": str(path),
        "bytes": len(raw),
        "encoding": encoding,
        "physical_lines": len(lines),
        "records": record_count,
        "parse_error_count": parse_error_count,
        "empty_field_count": empty_field_count,
        "parse_error_samples": parse_errors,
        "newlines": {"crlf": crlf_count, "lf": lf_count},
        "sentence_chars": distribution(sentence_lengths),
        "s1_chars": distribution(s1_lengths),
        "s2_chars": distribution(s2_lengths),
    }

    if expects_label:
        result["labels"] = {
            str(label): {
                "count": count,
                "ratio": round(count / record_count, 6),
            }
            for label, count in sorted(label_counts.items())
        }

    return result


def audit_overlaps(data_dir: Path) -> dict[str, object]:
    """Check duplicate sentence pairs within and across train/dev/test."""
    pair_labels: dict[str, dict[tuple[str, str], int]] = {}
    within_split_duplicates: dict[str, int] = {}
    within_split_label_conflicts: dict[str, int] = {}

    for split in SPLITS:
        labels: dict[tuple[str, str], int] = {}
        duplicates = 0
        conflicts = 0
        path = data_dir / f"{split}.txt"
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 3:
                continue
            pair = (fields[0], fields[1])
            label = int(fields[2])
            if pair in labels:
                duplicates += 1
                if labels[pair] != label:
                    conflicts += 1
            else:
                labels[pair] = label
        pair_labels[split] = labels
        within_split_duplicates[split] = duplicates
        within_split_label_conflicts[split] = conflicts

    cross_split: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            shared = pair_labels[left].keys() & pair_labels[right].keys()
            cross_split[f"{left}_{right}"] = {
                "overlap_pairs": len(shared),
                "label_conflicts": sum(
                    pair_labels[left][pair] != pair_labels[right][pair]
                    for pair in shared
                ),
            }

    return {
        "within_split_duplicate_pairs": within_split_duplicates,
        "within_split_label_conflicts": within_split_label_conflicts,
        "cross_split": cross_split,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect LCQMC TSV files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\Desktop\校\计算机\datasets\LCQMC"),
        help="Directory containing train.txt, dev.txt, and test.txt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = {}

    for split in SPLITS:
        path = args.data_dir / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        # The local LCQMC copy has a label column in all three files,
        # including test.txt; this is verified rather than assumed.
        report[split] = inspect_split(path, expects_label=True)
    report["overlap_audit"] = audit_overlaps(args.data_dir)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
