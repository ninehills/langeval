import json
import sys
from typing import List, Tuple

import tiktoken
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.faiss import FAISS

# embed = HuggingFaceEmbeddings(model_name="infgrad/stella-base-zh")
embed = OpenAIEmbeddings()
vectorstore = FAISS.load_local("cmrc-eval-zh.faiss", embed,
                               distance_strategy="COSINE")
def _retrieve_with_scores(query: str) -> List[Tuple[Document, float]]:
    # 自定义的原因是 VectorStoreRetriever 不返回 score
    docs =  vectorstore.similarity_search_with_relevance_scores(
        query, k=5, score_threshold=0.1)
    tokens = 0
    token_limit = 4000
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    for i, doc in enumerate(docs):
        tokens += len(enc.encode(doc[0].page_content))
        if tokens > token_limit:
            return docs[:i]
    return docs


inputs = json.loads(sys.stdin.read())
docs_with_score = _retrieve_with_scores(inputs["question"])
result_formatted = {
    "contexts": [i[0].page_content for i in docs_with_score],
    "contexts_scores": [i[1] for i in docs_with_score],
}

print(json.dumps(result_formatted, ensure_ascii=False, indent=2))
