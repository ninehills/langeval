import json
import sys
import xmlrpc.client
from typing import List, Tuple

import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS

s = xmlrpc.client.ServerProxy("http://localhost:8000")

embed = OpenAIEmbeddings()
vectorstore = FAISS.load_local("../cmrc-eval-zh.faiss", embed,
                               distance_strategy="COSINE")
def _retrieve_with_scores(query: str) -> List[Tuple[Document, float]]:
    # 自定义的原因是 VectorStoreRetriever 不返回 score
    # 使用reranker 需要扩大第一步的 Top K，然后reranker 二次筛选
    docs =  vectorstore.similarity_search_with_relevance_scores(
        query, k=30, score_threshold=0.0)
    return docs

def _rerank_with_scores(query: str, docs_with_score: List[Tuple[Document, float]]) -> List[Tuple[Document, float, float]]:
    rerank_scores = s.reranker_compute_score(query, [i[0].page_content for i in docs_with_score])
    ret = []
    for i, j in enumerate(docs_with_score):
        ret.append((j[0], j[1], rerank_scores[i])) # type: ignore
    ret = sorted(ret, key=lambda x: x[2], reverse=True)

    # 在ret 中寻找 TopK，并注意 token 数量限制
    topk = 5
    tokens = 0
    token_limit = 4000
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for i, d in enumerate(ret):
        if i >= topk:
            return ret[:i]
        tokens += len(enc.encode(d[0].page_content))
        if tokens > token_limit:
            return ret[:i]
    return ret


inputs = json.loads(sys.stdin.read())
docs_with_score = _retrieve_with_scores(inputs["question"])
docs_with_rerank_score = _rerank_with_scores(inputs["question"], docs_with_score)

result_formatted = {
    "contexts": [i[0].page_content for i in docs_with_rerank_score],
    "contexts_scores": [i[1] for i in docs_with_rerank_score],
    "contexts_rerank_scores": [i[2] for i in docs_with_rerank_score],
}

print(json.dumps(result_formatted, ensure_ascii=False, indent=2))
