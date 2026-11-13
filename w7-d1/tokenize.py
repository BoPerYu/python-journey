import argparse
import unicodedata
from pathlib import Path
import jieba


def _require_data_file(path):
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}\n")
    return path

def _normalize(text):
    text=unicodedata.normalize("NFKC",text)
    text=text.lower()
    return text

def _token_list(text):
    token_list=jieba.lcut(text)
    token_list=[token for token in token_list if any(ch.isalnum() for ch in token)]
    return token_list

def _file_working(path, first_column=False, limit=None):
    with path.open("r",encoding="UTF-8") as f:
        processed_lines=[]
        for line_number, raw_line in enumerate(f, start=1):
            if limit is not None and len(processed_lines) >= limit:
                break
            line=raw_line.rstrip("\r\n")
            if first_column:
                fields=line.split("\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"{path} 第 {line_number} 行不是三列 TSV："
                        f"实际 {len(fields)} 列"
                    )
                line=fields[0]
            tokens=_normalize(line)
            token_list=_token_list(tokens)
            output_line=" ".join(token_list)
            processed_lines.append(output_line)
    return processed_lines

def main():
    parser=argparse.ArgumentParser(
        description="Normalize and tokenize one document per line."
    )
    parser.add_argument("--input", type=Path, default=Path("input.txt"))
    parser.add_argument("--output", type=Path, default=Path("input.tokens.txt"))
    parser.add_argument(
        "--first-column",
        action="store_true",
        help="Read only the first field of three-column TSV input.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many documents.",
    )
    args=parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    input_path=args.input
    output_path=args.output
    _require_data_file(input_path)
    processed_line=_file_working(
        input_path,
        first_column=args.first_column,
        limit=args.limit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w",encoding="UTF-8",newline="\n") as f:
        for line in processed_line:
            f.write(line+"\n")

if __name__=="__main__":
    main()

