import enum
import logging
from typing import Any, Optional

import jinja2

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

from sklearn.metrics import ndcg_score

from langeval.evaluators.exception import EvalRunError
from langeval.evaluators.rag.utils import overlap_coefficient_contain
from langeval.models import LLM
from langeval.providers.output_parser import SimpleJsonOutputParser

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_SIMILARITY_THRESHOLD = 0.8
DEFAULT_ANSWER_CORRECTNESS_PROMPT = """
你是一名教师，接下来你需要根据【问题】和【参考答案】，来评价【学生答案】。

# 评分过程

1. 首先思考【问题】下，【参考答案】和【学生答案】的一致性。
2. 根据一致性，给出对【学生答案】的评价理由。
3. 根据评价理由，给出对【学生答案】的评分。（0代表错误，0.5代表部分正确，1代表完全正确）。

# 输入

问题："{{ question }}"
参考答案："{{ reference_answer }}"
学生答案："{{ answer }}"

# 输出

输出为 Markdown JSON 格式的字符串，示例：

```json
{
    "answer_correctness_reasoning": "<评价理由>",
    "answer_correctness": 0
}
```

注意分数只能为 0、0.5、1，0 代表错误，0.5 代表部分正确，2 代表完全正确。

输出（Markdown JSON 格式）：
"""

class RagMetric(str, enum.Enum):
    # Compare contexts and reference_context
    RetrievalRecall = "retrieval_recall"
    # Compare answer and reference_answer with LLM-JUDGE correctness
    AnswerCorrectness = "answer_correctness"


class Rag(pc.BaseModel):
    rag_metrics: list[RagMetric]
    rag_llm: Optional[LLM] = None
    answer_correctness_prompt: Optional[str] = None


def retrieval_recall(rag: Rag, kwargs: dict[str, Any], timeout, default_llm):   # noqa: ARG001
    contexts = kwargs["contexts"]
    reference_context = kwargs["reference_context"]
    if len(contexts) == 0:
        return {
            "retrieval_recall": 0.0,
            "retrieval_recall_hit_rate": 0.0,
            "retrieval_recall_mrr": 0.0
        }
    # 通过判断 reference_context 是否在 contexts 中，以及所在的位置来计算。
    # 这里使用 ndcg 算法：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html
    # 假设 reference_context 和 contexts 是同一批语料的不同切分形式，所以不需要语义相似度，而是比对文本相似度。
    # 使用修改后的 Overlap Coefficient 算法计算相似度。取值范围是 0-1
    # 这种算法，当 contexts 包含 reference_context 时，得分是 1。
    true_relevance: list[float] = []
    # 如下是假设 reference_context 和 contexts 是相同语料。
    hit_rate = 0.0 # 代表 reference_context 是否在 contexts 中精确匹配。
    mrr = 0.0 # 计算 reference_context 的精确匹配位置，首位为 1.0，次位为 0.5，以此类推。
    for i, context in enumerate(contexts):
        if reference_context == context:
            hit_rate = 1.0
            if mrr == 0:
                # if 是避免 contexts 中有重复内容。
                mrr = 1.0 / (i + 1)
        true_relevance.append(overlap_coefficient_contain(
            context, reference_context
        ))
    # scores 的绝对值没有意义，只要是倒序排列就行。
    scores = list(range(len(contexts), 0, -1))
    # k = 10 代表只统计前 10 个结果
    ndgc_10 = ndcg_score([true_relevance], [scores], k=10) # type: ignore

    return {
        "retrieval_recall": hit_rate,
        "retrieval_recall_mrr": mrr,
        "retrieval_recall_ndgc_10": ndgc_10,
    }

def answer_correctness(rag: Rag, kwargs: dict[str, Any], timeout, default_llm):
    if rag.rag_llm:
        llm = rag.rag_llm
    else:
        llm = default_llm
    if llm is None:
        raise EvalRunError("llm is None, can not eval answer_correctness")

    prompt_tpl = DEFAULT_ANSWER_CORRECTNESS_PROMPT \
        if rag.answer_correctness_prompt is None else rag.answer_correctness_prompt
    prompt = jinja2.Template(prompt_tpl).render(**kwargs)
    text = llm.completion(prompt, timeout=timeout)
    eval_result = {}
    eval_result = SimpleJsonOutputParser().parse(text)
    if "answer_correctness" not in eval_result:
        raise EvalRunError("eval completion result missing key: answer_correctness")
    return eval_result


metrics_eval_funcs = {
    RagMetric.RetrievalRecall: retrieval_recall,
    RagMetric.AnswerCorrectness: answer_correctness,
}

def eval_rag(rag: Rag, kwargs: dict[str, Any], timeout, default_llm) -> dict[str, Any]:
    """Rag eval"""
    eval_result = {}
    for metric in rag.rag_metrics:
        if metric not in metrics_eval_funcs:
            raise EvalRunError(f"eval rag metric not supported: {metric}")
        eval_func = metrics_eval_funcs[metric]
        r = eval_func(rag, kwargs, timeout, default_llm)
        eval_result.update(r)
    return eval_result
