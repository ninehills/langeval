import enum
import logging
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator

from langeval.models import LLM
from langeval.providers.exception import ProviderRunError

logger = logging.getLogger(__name__)


class ProviderType(str, enum.Enum):
    """Provider Type"""

    Completion = "completion"
    ChatCompletion = "chat_completion"
    Execute = "execute"


class ExecSettings(BaseModel):
    """Exec settings"""

    command: str = Field(..., min_length=1, max_length=1024)
    kwargs: dict = {}


class LLMSettings(BaseModel):
    """LLM settings"""

    llm: LLM
    prompt: str


class Provider(BaseModel):
    """Provider Config"""

    # provider types, completion, chat_completion, execute
    type: ProviderType  # noqa: A003
    input_variables: list[str]
    settings: Union[ExecSettings, LLMSettings]
    output_parser: str

    class Config:
        validate_assignment = True

    @validator("type")
    def type_must_be_valid(cls, v):  # noqa: N805
        if v not in [ProviderType.Completion, ProviderType.ChatCompletion, ProviderType.Execute]:
            raise ValueError("type must be one of completion, chat_completion, execute")
        return v

    @validator("output_parser")
    def output_parser_must_be_valid(cls, v):  # noqa: N805
        if v not in ["json", "text"]:
            raise ValueError("output_parser must be one of json, text")
        return v

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Optional["Provider"]:
        if yaml_str == "":
            return None
        try:
            return cls(**yaml.safe_load(yaml_str))
        except Exception as e:
            raise ValueError(f"Invalid yaml: {e}") from e

    def batch_call(self, inputs_list: list[dict[str, Any]], timeout: int):
        from langeval.providers.run import batch_call_exec, call_chat_completion, call_completion

        for key in self.input_variables:
            for inputs in inputs_list:
                if key not in inputs:
                    raise ProviderRunError(f"Missing input variable: {key}")
        if self.type == "completion":
            return [call_completion(self, inputs, timeout) for inputs in inputs_list]
        elif self.type == "chat_completion":
            return [call_chat_completion(self, inputs, timeout) for inputs in inputs_list]
        elif self.type == "execute":
            return batch_call_exec(self, inputs_list, timeout)
        else:
            raise ProviderRunError(f"Invalid type: {self.type}")