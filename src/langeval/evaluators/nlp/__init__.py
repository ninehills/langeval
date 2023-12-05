import enum
import logging
from typing import Any

import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

from rouge_chinese import Rouge

logger = logging.getLogger(__name__)

class NLPMetric(str, enum.Enum):
    # Exact Match
    ExactMatch = "exact_match"
    # Chinese BLEU
    BLEUChinese = "bleu_chinese"
    # Chinese Rouge
    RougeChinese = "rouge_chinese"


def exact_match(prediction: str, reference: str,
        stripped: bool = True, ignore_case: bool = False) -> dict[str, float]:
    """Exact Match

    Args:
        prediction (str): prediction
        reference (str): reference
        stripped (bool): remove leading and trailing whitespaces, default True
        ignore_case (bool): ignore case, default False

    Returns:
        dict[str, float]: {"exact_match": 0.0/1.0}
    """
    if stripped:
        prediction = prediction.strip()
        reference = reference.strip()
    if ignore_case:
        prediction = prediction.lower()
        reference = reference.lower()
    return {
        "exact_match": float(prediction == reference)
    }

def jieba_tokenizer(text: str) -> list[str]:
    """Jieba Tokenizer"""
    return list(jieba.cut(text))

def bleu_chinese(prediction, reference,
        segment: bool = True):
    """Chinese BLEU

    Args:
        prediction (str): prediction
        reference (str): reference
        segment (bool): jieba word segment, default True

    Returns:
        dict[str, float]: {"bleu-4": 0.0-1.0}
    """
    if segment:
        prediction = jieba_tokenizer(prediction)
        reference = jieba_tokenizer(reference)
    else:
        prediction = list[prediction]
        reference = list[reference]
    return {
        "bleu-4": sentence_bleu([reference], prediction, smoothing_function=SmoothingFunction().method3)
    }

def rouge_chinese(prediction, reference):
    """Chinese Rouge

    Args:
        prediction (str): prediction
        reference (str): reference

    Returns:
        dict[str, float]: {"rouge-1": 0.0-1.0, "rouge-2": 0.0-1.0, "rouge-l": 0.0-1.0}
    """
    prediction = jieba_tokenizer(prediction)
    reference = jieba_tokenizer(reference)
    if len(" ".join(prediction).split()) == 0 or len(" ".join(reference).split()) == 0:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    rouge = Rouge()
    scores = rouge.get_scores(" ".join(prediction), " ".join(reference))[0]
    return {
        "rouge-1": scores["rouge-1"]["f"], # type: ignore
        "rouge-2": scores["rouge-2"]["f"], # type: ignore
        "rouge-l": scores["rouge-l"]["f"], # type: ignore
    }


metrics_eval_funcs = {
    NLPMetric.ExactMatch: exact_match,
    NLPMetric.BLEUChinese: bleu_chinese,
    NLPMetric.RougeChinese: rouge_chinese,
}


class NLP(pc.BaseModel):
    prediction_key: str
    reference_key: str
    nlp_metrics: list[NLPMetric]
    nlp_metrics_kwargs: dict[NLPMetric, Any] = pc.Field(default_factory=dict)

    def call(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Evaluate call"""
        prediction = kwargs[self.prediction_key]
        reference = kwargs[self.reference_key]
        results = {}
        for metric in self.nlp_metrics:
            result = metrics_eval_funcs[metric](prediction, reference,
                **self.nlp_metrics_kwargs.get(metric, {}))
            results.update(result)
        return results
