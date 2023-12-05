"""Evaluator Types"""
import copy
import enum
import logging
from typing import Any, Optional, Union

import yaml

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

from langeval.evaluators.exception import EvalRunError
from langeval.evaluators.nlp import NLP
from langeval.evaluators.rag import Rag
from langeval.evaluators.sql import SQLEvaluator
from langeval.models import LLM, Embedding

logger = logging.getLogger(__name__)

def represent_multiline_text(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    else:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, represent_multiline_text)


class EvaluatorType(str, enum.Enum):
    """Evaluator Type"""

    LLM_GRADE = "LLM_GRADE"
    EMBEDDING_COS_SIM = "EMBEDDING_COS_SIM"
    PYTHON_CODE = "PYTHON_CODE"
    RAG = "RAG"
    # Some NLP metrics
    NLP = "NLP"
    # sql evaluator
    SQL = "SQL"


class LLMGrade(pc.BaseModel):
    prompt: str
    eval_keys: list[str]
    llm: Optional[LLM] = None


class EmbeddingCosSim(pc.BaseModel):
    pairs_keys: tuple[str, str]
    embedding: Embedding
    cos_sim_threshold: float


class PythonCode(pc.BaseModel):
    code: str


EvaluatorSettings = {
    EvaluatorType.LLM_GRADE: LLMGrade,
    EvaluatorType.EMBEDDING_COS_SIM: EmbeddingCosSim,
    EvaluatorType.PYTHON_CODE: PythonCode,
    EvaluatorType.RAG: Rag,
    EvaluatorType.NLP: NLP,
    EvaluatorType.SQL: SQLEvaluator,
}


class Evaluator(pc.BaseModel):
    """Evaluator"""
    # Name
    name: str
    # Type
    type: EvaluatorType  # noqa: A003
    # Detail config
    settings: Optional[Union[Rag, LLMGrade, EmbeddingCosSim, PythonCode, NLP, SQLEvaluator]] = None

    def to_yaml(self) -> str:
        return yaml.dump(self.dict(exclude_unset=True), encoding="utf-8", allow_unicode=True).decode("utf-8")

    @pc.validator("type")
    def type_must_be_valid(cls, v):  # noqa: N805
        if EvaluatorType(v) not in EvaluatorSettings.keys():
            raise ValueError(f"Invalid type: {v}")
        return v

    def batch_call(self, batch_inputs: list[dict[str, Any]], batch_outputs: list[dict[str, Any]], timeout=10,
                   default_llm=None) -> list[dict[str, Any]]:
        """Do batch eval"""
        from langeval.evaluators.rag import eval_rag
        from langeval.evaluators.run import eval_embedding_cos_sim, eval_llm_grade, eval_python_code
        kwargs_list = []
        for i, inputs in enumerate(batch_inputs):
            kwargs = copy.copy(inputs)
            kwargs.update(batch_outputs[i])
            kwargs_list.append(kwargs)

        results = []
        try:
            if self.type == EvaluatorType.LLM_GRADE:
                for kwargs in kwargs_list:
                    results.append(eval_llm_grade(self, kwargs, timeout, default_llm))
                return results
            elif self.type == EvaluatorType.EMBEDDING_COS_SIM:
                for kwargs in kwargs_list:
                    results.append(eval_embedding_cos_sim(self, kwargs, timeout))
                return results
            elif self.type == EvaluatorType.PYTHON_CODE:
                return eval_python_code(self, kwargs_list, timeout)
            elif self.type == EvaluatorType.RAG:
                if self.settings is None or not isinstance(self.settings, Rag):
                    raise EvalRunError(f"RAG settings is not specified: {self.settings}")
                for kwargs in kwargs_list:
                    results.append(eval_rag(self.settings, kwargs, timeout, default_llm))
                return results
            elif self.type == EvaluatorType.NLP:
                if self.settings is None or not isinstance(self.settings, NLP):
                    raise EvalRunError(f"NLP settings is not specified: {self.settings}")
                for kwargs in kwargs_list:
                    results.append(self.settings.call(kwargs))
                return results
            elif self.type == EvaluatorType.SQL:
                if self.settings is None or type(self.settings) != SQLEvaluator:
                    raise EvalRunError(f"SQL settings is not specified: {self.settings}")
                for kwargs in kwargs_list:
                    results.append(self.settings.call(kwargs, timeout))
                return results

        except Exception as e:
            logger.exception(f"eval failed: {e}")
            logger.debug(f"evaluator {self} eval failed: {e}", exc_info=True)
            raise EvalRunError(f"eval failed: {e}") from e

        raise EvalRunError(f"eval type not supported: {self.type}")
