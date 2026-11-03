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

def _file_working(path):
    with path.open("r",encoding="UTF-8") as f:
        processed_lines=[]
        for raw_line in f:
            line=raw_line.rstrip("\r\n")
            tokens=_normalize(line)
            token_list=_token_list(tokens)
            output_line=" ".join(token_list)
            processed_lines.append(output_line)
    return processed_lines

def main():
    input_path=Path("input.txt")
    output_path=Path("input.tokens.txt")
    _require_data_file(input_path)
    processed_line=_file_working(input_path)
    with output_path.open("w",encoding="UTF-8",newline="\n") as f:
        for line in processed_line:
            f.write(line+"\n")

if __name__=="__main__":
    main()

