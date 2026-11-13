from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import csv
import argparse


sentences = [
    "今天上海下雨吗？",
    "上海今天会下雨吗？",

    "怎么重置登录密码？",
    "忘记密码了应该怎么找回？",

    "这个手机多少钱？",
    "这台手机的价格是多少？",

    "我想申请退款。",
    "退款应该怎么操作？",

    "地铁几点停运？",
    "末班地铁是什么时候？",

    "苹果香蕉卖多少钱？",
    "水果的价格是多少？",

    "小猫喜欢喝牛奶。",
    "小狗喜欢啃骨头。",

    "今天温度很低。",
    "今天的天气非常冷。",

    "如何学习 Python？",
    "怎样学习 C++？",

    "高铁票怎么改签？",
    "飞机票如何退票？",
]

def analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args=parser.parse_args()
    return(args.model_dir,args.output_dir,args.batch_size)

def check(model_dir,output_dir,batch_size):
    model_dir_path=Path(model_dir)
    output_dir_path=Path(output_dir)
    if not model_dir_path.exists():
        raise FileNotFoundError ("模型不存在")
    if output_dir_path.exists() and not output_dir_path.is_dir():
        raise FileNotFoundError("输出文件不存在")
    if batch_size<=0:
        raise ValueError("batch_size 必须为正整数")
    return(model_dir_path,output_dir_path,batch_size)

def prepare_sentences(sentence: list[str]) -> list[str]:
    unique_sentences = list(dict.fromkeys(sentence))
    if len(unique_sentences)!=20:
        raise ValueError ("存在重复句子")
    return unique_sentences

def load_embedding_model(model_dir:Path)->SentenceTransformer:
    model = SentenceTransformer(
            str(model_dir),
            device="cpu",
        )
    return model

def encode_sentences(model: SentenceTransformer,texts: list[str],batch_size) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings

def compute_pair_scores(
    embeddings: np.ndarray,
    texts: list[str],
    pairs: list[tuple[int, int]],
) -> list[tuple[str, str, float]]:
    result=[]
    for i,j in pairs:
        score=float(np.dot(embeddings[i],embeddings[j]))
        text_a=texts[i]
        text_b=texts[j]
        result.append((text_a,text_b,score))
    return result

def save_embeddings(output_dir: Path, embeddings: np.ndarray)->None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path=output_dir / "embeddings.npy"
    np.save(output_path, embeddings)

def save_pair_scores(
    output_dir:Path,
    pairs: list[tuple[int, int]], 
    texts: list[str], 
    scores: list[tuple[str, str, float]]):
    with (output_dir / "pair_scores.csv").open("w",encoding="utf-8",newline="")as f:
        writer=csv.writer(f)
        writer.writerow(["index_a","index_b","sentence_a","sentence_b","cosine"])
        for i,(sentence_a,sentence_b,cosine) in enumerate(scores):
            index_a,index_b=pairs[i]
            writer.writerow([index_a, index_b, sentence_a, sentence_b, f"{cosine:.4f}"])

def main():
    model_dir,output_dir,batch_size=analysis()
    model_dir=Path(model_dir)
    output_dir=Path(output_dir)
    batch_size=batch_size
    check(model_dir,output_dir,batch_size)

    pairs=[(0,1),(2,3),(8,9),(12,13),(16,17)]
    text=prepare_sentences(sentences)
    model=load_embedding_model(model_dir)
    embeddings=encode_sentences(model, text, batch_size)
    result=compute_pair_scores(embeddings,text,pairs)

    assert embeddings.shape == (20, 1024)
    norms = np.linalg.norm(embeddings, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    print("max norm error:", max_norm_error)
    print("norm range:", float(norms.min()), float(norms.max()))
    assert max_norm_error < 1e-5
    assert np.isfinite(embeddings).all()

    save_embeddings(output_dir, embeddings)
    save_pair_scores(output_dir,pairs, text,result)

    print("embedding shape:", embeddings.shape)
    for index_a, index_b in pairs:
        print(f"{index_a:02d} vs {index_b:02d}: "
              f"{float(np.dot(embeddings[index_a], embeddings[index_b])):.6f}")

if __name__=="__main__":
    main()
