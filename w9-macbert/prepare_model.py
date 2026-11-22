"""只下载本实验需要的 MacBERT-base 文件，不改动用户已有 large 模型。"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import urllib.request

REPO = "hfl/chinese-macbert-base"
REVISION = "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
FILES = ("config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "added_tokens.json", "pytorch_model.bin")
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "models" / "chinese-macbert-base"


def download(name, target):
    partial = target.with_name(target.name + ".part")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                resume = response.status == 206 and offset > 0
                if not resume:
                    offset = 0
                expected = response.headers.get("Content-Length")
                received = 0
                next_progress = offset + 32 * 1024 * 1024
                with partial.open("ab" if resume else "wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                        received += len(chunk)
                        if offset + received >= next_progress:
                            print(f"{name}: {(offset + received) / 1024**2:.0f} MiB", flush=True)
                            next_progress += 32 * 1024 * 1024
                if expected is not None and received != int(expected):
                    raise OSError("下载长度与服务器声明不符")
            if name.endswith(".json"):
                json.loads(partial.read_text(encoding="utf-8"))
            if name.endswith(".bin") and partial.stat().st_size < 100_000_000:
                raise OSError("权重文件过小，可能没有取得实际模型")
            partial.replace(target)
            return
        except Exception as exc:
            print(f"download retry {attempt + 1}/3: {name}: {exc}", flush=True)
            if attempt == 2:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = args.model_dir / name
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            download(name, target)
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    if (config["num_hidden_layers"], config["hidden_size"]) != (12, 768):
        raise ValueError("本实验要求 base 的 12 层、768 维配置")
    fingerprints = {}
    for name in FILES:
        with (args.model_dir / name).open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        fingerprints[name] = {"bytes": (args.model_dir / name).stat().st_size, "sha256": digest}
    provenance = {"repo": REPO, "revision": REVISION, "files": fingerprints}
    (args.model_dir / "download_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MODEL_READY {args.model_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
