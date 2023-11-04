"""Evaluator Types"""
import enum
import logging
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, validator

from langeval.evaluators.exception import EvalRunError
from langeval.evaluators.rag import Rag
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


class LLMGrade(BaseModel):
    prompt: str
    llm: Optional[LLM] = None


class EmbeddingCosSim(BaseModel):
    embedding: Embedding
    cos_sim_threshold: float


class PythonCode(BaseModel):
    code: str


EvaluatorSettings = {
    EvaluatorType.LLM_GRADE: LLMGrade,
    EvaluatorType.EMBEDDING_COS_SIM: EmbeddingCosSim,
    EvaluatorType.PYTHON_CODE: PythonCode,
    EvaluatorType.RAG: Rag,
}


class Evaluator(BaseModel):
    """Evaluator"""

    # Provider input keys
    input_keys: list[str] = []
    # Provider output keys
    output_keys: list[str] = []
    # Eval output keys
    eval_keys: list[str] = []
    # Type
    type: EvaluatorType  # noqa: A003
    # Detail config
    settings: Optional[Union[Rag, LLMGrade, EmbeddingCosSim, PythonCode]] = None

    class Config:
        validate_assignment = True

    def to_yaml(self) -> str:
        return yaml.dump(self.dict(exclude_unset=True), encoding="utf-8", allow_unicode=True).decode("utf-8")

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Optional["Evaluator"]:
        if yaml_str == "":
            return None

        obj = yaml.load(yaml_str, Loader=yaml.FullLoader)  # noqa: S506
        return cls(**obj)

    @validator("type")
    def type_must_be_valid(cls, v):  # noqa: N805
        if EvaluatorType(v) not in EvaluatorSettings.keys():
            raise ValueError(f"Invalid type: {v}")
        return v

    def call(self, inputs: dict[str, Any], outputs: dict[str, Any], timeout=10, default_llm=None) -> dict[str, Any]:
        """Do eval"""
        from langeval.evaluators.rag import eval_rag
        from langeval.evaluators.run import eval_embedding_cos_sim, eval_llm_grade, eval_python_code
        kwargs = {}
        for k in self.input_keys:
            if k not in inputs:
                msg = f"eval input missing key: {k}"
                raise EvalRunError(msg)
            kwargs[k] = inputs[k]

        for k in self.output_keys:
            if k in outputs:
                kwargs[k] = outputs[k]
            elif k in inputs:
                kwargs[k] = inputs[k]
            else:
                msg = f"eval input missing key: {k}"
                raise EvalRunError(msg)

        try:
            if self.type == EvaluatorType.LLM_GRADE:
                return eval_llm_grade(self, kwargs, timeout, default_llm)
            elif self.type == EvaluatorType.EMBEDDING_COS_SIM:
                return eval_embedding_cos_sim(self, kwargs, timeout)
            elif self.type == EvaluatorType.PYTHON_CODE:
                return eval_python_code(self, kwargs, timeout)
            elif self.type == EvaluatorType.RAG:
                if self.settings is None or type(self.settings) != Rag:
                    raise EvalRunError(f"RAG settings is not specified: {self.settings}")
                return eval_rag(self.settings, kwargs, timeout, default_llm)
        except Exception as e:
            logger.exception(f"eval failed: {e}")
            logger.debug(f"evaluator {self} eval failed: {e}", exc_info=True)
            raise EvalRunError(f"eval failed: {e}") from e

        raise EvalRunError(f"eval type not supported: {self.type}")
