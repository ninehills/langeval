from langeval.evaluators.evaluator import EmbeddingCosSim, Evaluator, EvaluatorType, LLMGrade, PythonCode
from langeval.models import LLM, Embedding

LLM_QA_GRADE_PROMPT_TPL = """你是一名教师，接下来你需要根据【问题】和【参考答案】，来评价【学生答案】。

# 评分过程

1. 首先思考【问题】下，【参考答案】和【学生答案】的一致性。
2. 根据一致性，给出对【学生答案】的评价理由。
3. 根据评价理由，给出对【学生答案】的评分。（0代表错误，0.5代表部分正确，1代表完全正确）。

# 输入

问题："{{ query }}"
参考答案："{{ answer }}"
学生答案："{{ text }}"

# 输出

输出为 Markdown JSON 格式的字符串，示例：

```json
{
    "reasoning": "<评价理由>",
    "score": 0
}
```

注意分数只能为 0、0.5、1，0 代表错误，0.5 代表部分正确，2 代表完全正确。

输出（Markdown JSON 格式）：
"""

ExampleLLMGradeEvaluatorChinese = Evaluator(
    name="llm_qa_grade",
    type=EvaluatorType.LLM_GRADE,
    input_keys=["query", "answer"],
    output_keys=["text"],
    eval_keys=["reasoning", "score"],
    settings=LLMGrade(
        prompt=LLM_QA_GRADE_PROMPT_TPL,
        llm=LLM(
            provider="openai",
            model="gpt-3.5-turbo",
            kwargs={
                "temperature": 0.1,
            },
        ),
    ),
)

ExampleEmbeddingCosSimEvaluator = Evaluator(
    name="cos_sim",
    type=EvaluatorType.EMBEDDING_COS_SIM,
    input_keys=["answer"],
    output_keys=["text"],
    eval_keys=["cos_sim", "is_similar"],
    settings=EmbeddingCosSim(
        embedding=Embedding(
            provider="openai",
            model="text-embedding-ada-002",
            kwargs={},
        ),
        cos_sim_threshold=0.7,
    ),
)

ExamplePythonCodeEvaluator = Evaluator(
    name="exact_match",
    type=EvaluatorType.PYTHON_CODE,
    input_keys=["answer"],
    output_keys=["text"],
    eval_keys=["score"],
    settings=PythonCode(
        code="""import json
import sys
kwargs_list = json.load(sys.stdin)
results = []
for kwargs in kwargs_list:
    results.append(dict(score=(int(kwargs["answer"] == kwargs["text"]))))
print(json.dumps(results))
"""
    ),
)
