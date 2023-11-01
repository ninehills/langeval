import logging

from pydantic import BaseModel, validator

from langeval.models.exception import ModelRunError

logger = logging.getLogger(__name__)


class LLM(BaseModel):
    provider: str
    model: str
    # Model parameters, e.g. Qianfan has ak, sk, etc.
    kwargs: dict = {}

    @validator("provider")
    def provider_must_be_valid(cls, v):  # noqa: N805
        if v not in ["qianfan", "openai", "langchain"]:
            raise ValueError(f"Invalid provider: {v}")
        return v

    def completion(self, prompt: str, timeout: int = 10) -> str:
        """Generate completion for prompt"""
        if self.provider == "qianfan":
            return call_qianfan(self.model, self.kwargs, prompt, [], timeout)
        elif self.provider == "openai":
            return call_openai(self.model, self.kwargs, prompt, [], timeout)
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
            return call_qianfan(self.model, self.kwargs, "", messages, timeout)
        elif self.provider == "openai":
            return call_openai(self.model, self.kwargs, "", messages, timeout)
        elif self.provider == "langchain":
            raise ValueError("langchain does not support chat_model load yet")
        else:
            raise ValueError(f"Invalid provider: {self.provider}")


def call_qianfan(model: str, kwargs: dict, prompt: str, messages: list, timeout: int) -> str:
    try:
        import qianfan
        import qianfan.errors
    except ImportError as e:
        raise ValueError(
            "Could not import qianfan python package. Please install it with `pip install qianfan`."
        ) from e
    try:
        if prompt:
            client = qianfan.Completion(model=model, **kwargs)
            res = client.do(prompt, request_timeout=float(timeout))
        else:
            client = qianfan.ChatCompletion(model=model, **kwargs)
            res = client.do(messages, request_timeout=float(timeout))
        if res.code != 200:  # type: ignore # noqa: PLR2004
            raise ModelRunError(f"qianfan call failed: {res}")
        result = res.body.get("result", None)  # type: ignore
        if not result:
            raise ModelRunError(f"qianfan call failed: {res}")
        return result
    except qianfan.errors.QianfanError as e:
        raise ModelRunError(f"qianfan call failed: {e.__class__.__name__}({e})") from e


def call_openai(model: str, kwargs: dict, prompt: str, messages: list, timeout: int = 30) -> str:
    try:
        import openai
        import openai.error
    except ImportError as e:
        raise ValueError("Could not import openai python package. Please install it with `pip install openai`.") from e
    try:
        if not model.endswith("-instruct") and prompt:
            # When chat model use prompt, then convert it to messages
            messages = [{"role": "user", "content": prompt}]
        if messages:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                request_timeout=float(timeout),
                **kwargs,
            )
            logger.debug(f"openai completion: {messages} -> {response}")
            return response.choices[0].message["content"]  # type: ignore
        else:
            kwargs = kwargs.copy()
            if "max_tokens" not in kwargs:
                # Default to 1024 tokens
                kwargs["max_tokens"] = 1024
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                request_timeout=float(timeout),
                **kwargs,
            )
            logger.debug(f"openai completion: {prompt} -> {response}")
            return response.choices[0].text  # type: ignore
    except openai.error.OpenAIError as e:
        raise ModelRunError(f"openai call failed: {e.__class__.__name__}({e})") from e
