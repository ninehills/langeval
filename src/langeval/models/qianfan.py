try:
    import qianfan
    import qianfan.errors
except ImportError as e:
    raise ValueError(
        "Could not import qianfan python package. Please install it with `pip install qianfan`."
    ) from e

from typing import Any

from langeval.models.exception import ModelRunError


class QianfanChatCompletion:
    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs
        self.client = qianfan.ChatCompletion(model=model, **kwargs)

    def call(self, prompt: str, messages: list, timeout: int) -> str:
        try:
            kwargs = {}
            if prompt:
                messages_converted = [{"role": "user", "content": prompt}]
            else:
                system = ""
                messages_converted = []
                for message in messages:
                    if message["role"] == "system":
                        system = message["content"]
                        continue
                    messages_converted.append(message)
                if system:
                    kwargs["system"] = system
            res = self.client.do(
                messages_converted, request_timeout=float(timeout), **kwargs)
            if res.code != 200:  # type: ignore # noqa: PLR2004
                raise ModelRunError(f"qianfan call failed: {res}")
            result = res.body.get("result", None)  # type: ignore
            if not result:
                raise ModelRunError(f"qianfan call failed: {res}")
            return result
        except qianfan.errors.QianfanError as e:
            raise ModelRunError(f"qianfan call failed: {e.__class__.__name__}({e})") from e


class QianfanCompletion:
    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.kwargs = kwargs
        self.client = qianfan.Completion(model=model, **kwargs)

    def call(self, prompt: str, timeout: int) -> str:
        try:
            res = self.client.do(prompt, request_timeout=float(timeout))
            if res.code != 200:  # type: ignore # noqa: PLR2004
                raise ModelRunError(f"qianfan call failed: {res}")
            result = res.body.get("result", None)  # type: ignore
            if not result:
                raise ModelRunError(f"qianfan call failed: {res}")
            return result
        except qianfan.errors.QianfanError as e:
            raise ModelRunError(f"qianfan call failed: {e.__class__.__name__}({e})") from e
