# Rerank

```bash
pip install -U FlagEmbedding protobuf
```

Running in GPU is recommended

观察 reranker score，可以发现比embedding cos_sim 更明显的区分度。
一个简单观察，可以用 reranker score 大于0和小于0来区分是否语义相似。过滤掉小于0的，应该效果会更好，反而是 embedding 的score 过滤没有什么意义。

start server（使用 server是为了避免在 eval 时重复加载模型，加快速度）
server 使用简单的 XMLRPC 实现。

```bash
python3 retrieval-rerank-server.py
```

run eval

```bash
export OPENAI_API_KEY="xxx"
langeval -vv run ./retrieval-rerank-eval.yaml
```

测试结果:

- "retrieval_recall": 0.9060214999 -> 0.9754992113
- "retrieval_recall_hit_rate": 0.91 -> 0.97
- "retrieval_recall_mrr": 0.8853333333 -> 0.9575

hit_rate 大幅提升到 6%。
