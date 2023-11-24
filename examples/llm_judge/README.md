# `ceval-llm-judge` 任务，使用 GPT-4 进行判卷。

```bash
export OPENAI_API_KEY="sk-xxxxx"
# 单个问题
langeval run ceval-llm-judge.yaml --sample 1

# 全量测试
langeval run ceval-llm-judge.yaml
```
