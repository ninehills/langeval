import logging
from typing import Any

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

from langeval.models.exception import ModelRunError
from langeval.models.openai import OpenAI
from langeval.models.qianfan import Qianfan

logger = logging.getLogger(__name__)


class LLM(pc.BaseModel):
    provider: str
    model: str
    # Model parameters, e.g. Qianfan has ak, sk, etc.
    kwargs: dict = {}
    instance: Any = None

    @pc.validator("provider")
    def provider_must_be_valid(cls, v):  # noqa: N805
        if v not in ["qianfan", "openai", "langchain"]:
            raise ValueError(f"Invalid provider: {v}")
        return v

    def completion(self, prompt: str, timeout: int = 10) -> str:
        """Generate completion for prompt"""
        if self.provider == "qianfan":
            if self.instance is None:
                self.instance = Qianfan(self.model)
            return self.instance.call(prompt, [], timeout, **self.kwargs)
        elif self.provider == "openai":
            if self.instance is None:
                self.instance = OpenAI(self.model)
            return self.instance.call(prompt, [], timeout, **self.kwargs)
        elif self.provider == "langchain":
            try:
                from langchain.llms.loading import load_llm_from_config

                llm = load_llm_from_config(
                    dict(
                        _type=self.provider,
                        model_name=self.model,
                        **self.kwargs,
                    )
                )
            except ImportError as e:
                raise ValueError(
                    "Could not import langchain python package or llm package."
                    "Please install it with `pip install langchain`."
                ) from e
            try:
                response = llm.predict(prompt, request_timeout=float(timeout))
                logger.debug(f"langchain completion: {prompt} -> {response}")
                return response
            except Exception as e:
                raise ModelRunError(f"langchain call failed: {e.__class__.__name__}({e})") from e
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def chat_completion(self, messages: list[dict[str, str]], timeout: int = 10) -> str:
        """Generate chat completion for messages"""
        if self.provider == "qianfan":
            if self.instance is None:
                self.instance = Qianfan(self.model)
            return self.instance.call("", messages, timeout, **self.kwargs)
        elif self.provider == "openai":
            if self.instance is None:
                self.instance = OpenAI(self.model)
            return self.instance.call("", messages, timeout, **self.kwargs)
        elif self.provider == "langchain":
            raise ValueError("langchain does not support chat_model load yet")
        else:
            raise ValueError(f"Invalid provider: {self.provider}")
