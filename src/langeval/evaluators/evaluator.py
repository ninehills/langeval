"""Evaluator Types"""
import enum
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, validator

from langeval.evaluators.exception import EvalRunError
from langeval.models import LLM, Embedding


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


EvaluatorTypeList = list(EvaluatorType)


class LLMGrade(BaseModel):
    prompt: str
    llm: Optional[LLM] = None


class EmbeddingCosSim(BaseModel):
    embedding: Embedding
    cos_sim_threshold: float = 0.9


class PythonCode(BaseModel):
    code: str


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
    settings: Optional[Union[LLMGrade, EmbeddingCosSim, PythonCode]] = None

    class Config:
        validate_assignment = True

    def to_yaml(self) -> str:
        return yaml.dump(self.dict(exclude_unset=True), encoding="utf-8", allow_unicode=True).decode("utf-8")

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Optional["Evaluator"]:
        if yaml_str != "":
            return None
        else:
            obj = yaml.load(yaml_str, Loader=yaml.FullLoader)  # noqa: S506
            settings = None
            if obj["type"] == str(EvaluatorType.LLM_GRADE):
                settings = LLMGrade(**obj["settings"])
            elif obj["type"] == str(EvaluatorType.EMBEDDING_COS_SIM):
                settings = EmbeddingCosSim(**obj["settings"])
            elif obj["type"] == str(EvaluatorType.PYTHON_CODE):
                settings = PythonCode(**obj["settings"])

            obj["settings"] = settings
        return cls(**obj)

    @validator("type")
    def type_must_be_valid(cls, v):  # noqa: N805
        if EvaluatorType(v) not in EvaluatorTypeList:
            raise ValueError(f"Invalid type: {v}")
        return v

    def call(self, inputs: dict[str, Any], outputs: dict[str, Any], timeout=10, default_llm=None) -> dict[str, Any]:
        """Do eval"""
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

        if self.type == EvaluatorType.LLM_GRADE:
            return eval_llm_grade(self, kwargs, timeout, default_llm)
        elif self.type == EvaluatorType.EMBEDDING_COS_SIM:
            return eval_embedding_cos_sim(self, kwargs, timeout)
        elif self.type == EvaluatorType.PYTHON_CODE:
            return eval_python_code(self, kwargs, timeout)
        else:
            msg = f"eval type not supported: {self.type}"
            raise EvalRunError(msg)
