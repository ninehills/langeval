import json
import sys
from operator import itemgetter
from typing import List, Tuple

import tiktoken
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
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

template = """仅使用如下上下文回答问题：
```
{context}
```

问题：{question}
回答：
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

def _combine_documents(docs_with_scores: List[Tuple[Document, float]]):
    return "\n\n".join([i[0].page_content for i in docs_with_scores])

_inputs = RunnablePassthrough()

retrieved_documents = {
    "docs_with_score": lambda x: _retrieve_with_scores(x["question"]),
    "question": itemgetter("question"),
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs_with_score"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | prompt | model,
    "docs_with_score": itemgetter("docs_with_score"),
}

final_chain = _inputs | retrieved_documents | answer

final_results = []
for i in json.loads(sys.stdin.read()):
    inputs = {
        "question": i["question"]
    }
    result = final_chain.invoke(inputs)
    final_results.append({
        "answer": result["answer"].content,
        "contexts": [i[0].page_content for i in result["docs_with_score"]],
        "contexts_scores": [i[1] for i in result["docs_with_score"]],
    })
print(json.dumps(final_results, ensure_ascii=False, indent=2))
