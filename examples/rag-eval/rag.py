import json
import sys
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.faiss import FAISS

embed = HuggingFaceEmbeddings(model_name="infgrad/stella-base-zh")
vectorstore = FAISS.load_local("cmrc-eval-zh.faiss", embed)
retriever = vectorstore.as_retriever()

template = """仅使用如下上下文回答问题：
```
{context}
```

问题：{question}
回答：
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

def _combine_documents(docs):
    return "\n\n".join([i.page_content for i in docs])

_inputs = RunnablePassthrough()

retrieved_documents = {
    "docs": itemgetter("question") | retriever,
    "question": itemgetter("question"),
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | prompt | model,
    "docs": itemgetter("docs"),
}

final_chain = _inputs | retrieved_documents | answer

inputs = {
    "question": json.loads(sys.stdin.read())["question"]
}
result = final_chain.invoke(inputs)
result_formatted = {
    "answer": result["answer"].content,
    "contexts": [i.page_content for i in result["docs"]],
}
print(json.dumps(result_formatted, ensure_ascii=False, indent=2))
