import enum
import logging
from typing import Any, Optional, Union

import yaml

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc


from langeval.models import LLM
from langeval.providers.exception import ProviderRunError
from langeval.providers.output_parser import OutputParserError, SimpleJsonOutputParser, SQLParser

logger = logging.getLogger(__name__)


class ProviderType(str, enum.Enum):
    """Provider Type"""

    Completion = "completion"
    ChatCompletion = "chat_completion"
    Execute = "execute"


class ExecSettings(pc.BaseModel):
    """Exec settings"""

    command: str = pc.Field(..., min_length=1, max_length=1024)
    kwargs: dict = {}


class LLMSettings(pc.BaseModel):
    """LLM settings"""

    llm: LLM
    prompt: str


class OutputParser(pc.BaseModel):
    """Output Parser"""

    name: str = pc.Field(default="text")
    # json parser:
    #    output_keys defines the keys to be extracted from the json output
    #    if not defined, all keys will be extracted
    # match parser:
    #    match_key: the key to be matched
    #    match_re: the regex to be matched
    kwargs: dict = {}


    def parse(self, text: str) -> dict[str, Any]:
        if self.name == "text":
            return {"text": text}
        elif self.name == "sql":
            try:
                resp = SQLParser().parse(text)
                if not resp:
                    raise ProviderRunError(f"output parser sql failed {text} -> {resp}")
            except OutputParserError as e:
                raise ProviderRunError(f"output parser sql failed {text}: {e}") from e
            return resp
        elif self.name == "json":
            try:
                resp = SimpleJsonOutputParser().parse(text)
                if not resp:
                    raise ProviderRunError(f"output parser failed {text} -> {resp}")
            except OutputParserError as e:
                raise ProviderRunError(f"output parser failed {text}: {e}") from e
            keys = self.kwargs.get("output_keys", None)
            final_resp = {}
            if keys:
                for key in keys:
                    if key not in resp:
                        raise ProviderRunError(f"output parser failed lack keys: {text} -> {resp}")
                    final_resp[key] = resp[key]
            final_resp["_text"] = text
            return final_resp
        elif self.name == "match":
            match_key = self.kwargs.get("match_key", None)
            match_re = self.kwargs.get("match_re", None)
            if not match_key or not match_re:
                raise ProviderRunError(f"Invalid output parser: {self.name} kwargs: {self.kwargs}")
            import re
            match = re.search(match_re.strip(), text)
            if not match:
                raise ProviderRunError(f"output parser failed: {text} match '{match_re.strip()}' failed")
            return {match_key: match.group(1), "_text": text}
        else:
            raise ProviderRunError(f"Invalid output parser: {self.name}")


class Provider(pc.BaseModel):
    """Provider Config"""

    # provider types, completion, chat_completion, execute
    type: ProviderType  # noqa: A003
    input_variables: list[str]
    settings: Union[ExecSettings, LLMSettings]
    output_parser: OutputParser

    class Config:
        validate_assignment = True

    @pc.validator("type")
    def type_must_be_valid(cls, v):  # noqa: N805
        if v not in [ProviderType.Completion, ProviderType.ChatCompletion, ProviderType.Execute]:
            raise ValueError("type must be one of completion, chat_completion, execute")
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
