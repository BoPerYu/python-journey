from pathlib import Path
from collections import Counter,defaultdict
import math

def load_documents(path) -> list[list[str]]:
    documents=[]
    with path.open("r",encoding="utf-8")as f:
        for i,line in enumerate(f,start=1):
            documents.append(line.split())
    return documents

def count_terms(tokens: list[str]) -> Counter:
    token_count_list=Counter()
    for token in tokens:
        token_count_list[token]+=1
    return token_count_list

def build_document_lengths(documents) -> list[int]:
    document_lengths=[0]*len(documents)
    for i,document in enumerate(documents):
        document_lengths[i]+=len(document)
    return document_lengths

def build_postings(documents) -> dict[str, list[tuple[int, int]]]:
    posting=defaultdict(list)
    for doc_id,document in enumerate(documents):
        token_tf=count_terms(document)
        for key,key_count in token_tf.items():
            posting[key].append((doc_id,key_count))
    return posting
        
def tfidf_idf(document_count: int, document_frequency: int) -> float:
    idf=float(math.log((document_count+1)/(document_frequency+1))+1)
    return idf

def build_tfidf_norms(postings, document_count) -> list[float]:
    norms=[0.0]*document_count
    for key,key_count_list in postings.items():
        key_df=len(key_count_list)
        key_idf=tfidf_idf(document_count,key_df)
        for doc_id,key_count in key_count_list:
            weight=float(key_idf*key_count)
            norms[doc_id]+=weight*weight
    for i in range(len(norms)):
        norms[i]=float(math.sqrt(norms[i]))
    return norms

def search_tfidf_from_index(
    doc_postings,
    doc_norms: list[float],
    document_count: int,
    query: str,
    top_k: int,
) -> list[tuple[int, float]]:
    if top_k <= 0:
        return []

    query_tokens=query.split()
    query_tf=count_terms(query_tokens)
    scores=[0.0]*document_count
    query_norm_squared=0.0
    for term,qtf in query_tf.items():
        if term not in doc_postings:
            continue
        term_posting=doc_postings[term]
        term_df=len(term_posting)
        term_idf=tfidf_idf(document_count,term_df)
        query_weight=qtf*term_idf
        query_norm_squared+=query_weight*query_weight
        for doc_id,dtf in term_posting:
            document_weight=dtf*term_idf
            scores[doc_id]+=query_weight*document_weight
    query_norm=math.sqrt(query_norm_squared)
    if query_norm==0:
        return []
    results=[]
    for doc_id in range(document_count):
        if scores[doc_id]==0 or doc_norms[doc_id]==0:
            continue
        score=scores[doc_id]/(query_norm*doc_norms[doc_id])
        results.append((doc_id,score))
    results.sort(key=lambda x:(-x[1],x[0]))
    return results[:top_k]


def search_tfidf(documents, query: str, top_k: int) -> list[tuple[int, float]]:
    doc_postings=build_postings(documents)
    doc_norms=build_tfidf_norms(doc_postings,len(documents))
    return search_tfidf_from_index(
        doc_postings=doc_postings,
        doc_norms=doc_norms,
        document_count=len(documents),
        query=query,
        top_k=top_k,
    )

