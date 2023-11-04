# Embedding 模型在 RAG 场景下的评估和微调

同时发布在：<https://github.com/ninehills/blog/issues/104>

为检验 Embedding 模型在 RAG 应用中的性能，我们引入 [C-MTEB](https://github.com/FlagOpen/FlagEmbedding/blob/master/C_MTEB/README.md) 评测用来评估 Embedding 模型的性能。

已有的 Embedding 模型的 C-MTEB 分数在 [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) 上可以通过选择 `Chinese` 选项卡查看。

而针对有明确数据集的场景，我们也可以复用 C-MTEB 的评估方法，评估 Embedding 模型在特定数据集上的性能，从而为后续微调提供参考。

## C-MTEB 评估任务

C-MTEB 有多种任务，其中和 RAG 能力相关是 Reranking 和 Retrieval 任务，其数据集格式如下：

**Reranking** 任务的数据集格式为：

```json
{
    "query": "大学怎么网上选宿舍",
    "positive": ["long text", "long text"],
    "negative": ["long text", "long text"]
}
```

目前通用场景的 Reranking 数据集主要是 [T2Reranking](https://huggingface.co/datasets/C-MTEB/T2Reranking)。

在评测分数中，我们主要关心 `map` 分数，这是因为任务不涉及排序，而是看是否命中 positive 。

而 **Retrieval** 任务的数据集以 [T2Retrieval](https://huggingface.co/datasets/C-MTEB/T2Retrieval) 为例，分为三个部分：

- corpus：是一个长文本集合。
- queries：是检索用的问题。
- qrels: 将检索问题和长文本对应起来，通过 score 进行排序。不过目前数据集中的 score 都为 1。（这可能是规避了数据错误标注，但是也影响了评测效果）。

在评测分数中，我们主要关心 `ndcg@10` 分数，是检验 top10 检索结果中排序是否一致的指标。

此外由于 Retrieval 数据集比较难构造，所以一般自定义数据集都是用 Reranking 数据集。Reranking 数据集的格式还和 [FlagEmbedding fine-tune](https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/README.md) 所需的数据格式相同，方便用于微调后的评估。

## 自定义模型的通用任务评测

选择 T2Reranking 进行评测，评测目标是文心千帆上的 [Embedding-V1](https://cloud.baidu.com/doc/WENXINWORKSHOP/s/alj562vvu) 模型。

参见 [Colab Notebook](https://colab.research.google.com/drive/1PcJcgWZ-B5AQUZ2FsRYd6inQ42_NqnUr?usp=sharing)。

取 1000 条测试数据（总数据的 1/6，为了降低 token 使用），评测 map 得分为 66.54。超过了 Leaderboard 上的 SOTA 分数 66.46 分。（不过并不是全部数据，这个分数仅供参考，如果是全部数据，得分可能会低于 SOTA）

## 自定义数据集微调和评测

使用 T2Reranking 数据集拆分出训练集和测试集，对 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) 模型进行微调和测试。

参见 [Colab Notebook](https://colab.research.google.com/drive/1dAAVssdWNin47e2xeGsEpWnArU6Nx4eu?usp=sharing)。

可以看到微调效果不尽如人意（第一个 checkpoint 就提升了 3% 的效果，然后后续无明显提升）。这可能是因为数据集的质量不高，导致模型无法学到有效的信息。

社区相关讨论：<https://github.com/FlagOpen/FlagEmbedding/issues/179>

微调资源占用：small 模型，4090 显存占用 12G。
