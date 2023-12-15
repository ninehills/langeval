try:
    import qianfan
    import qianfan.errors
except ImportError as e:
    raise ValueError(
        "Could not import qianfan python package. Please install it with `pip install qianfan`."
    ) from e

from typing import Any, List

from langeval.models.exception import ModelRunError
from langeval.models.types import Message


class Qianfan:
    def __init__(self, model: str, chat: bool = True):
        self.model = model
        self.chat = chat
        if chat:
            self.client = qianfan.ChatCompletion(model=model)
        else:
            self.client = qianfan.Completion(model=model)

    def call(self, prompt: str, messages: List[Message], timeout: int, **kwargs: Any) -> str:
        try:
            if prompt:
                messages_converted = [{"role": "user", "content": prompt}]
            else:
                system = ""
                messages_converted = []
                for message in messages:
                    if message.role == "system":
                        system = message.content
                        continue
                    messages_converted.append({"role": message.role, "content": message.content})
                if system:
                    kwargs["system"] = system
            if isinstance(self.client, qianfan.ChatCompletion):
                res = self.client.do(
                    messages_converted, request_timeout=float(timeout), **kwargs)
            else:
                res = self.client.do(
                    prompt, request_timeout=float(timeout), **kwargs)
            if res.code != 200:  # type: ignore # noqa: PLR2004
                raise ModelRunError(f"qianfan call failed: {res}")
            result = res.body.get("result", None)  # type: ignore
            if not result:
                raise ModelRunError(f"qianfan call failed: {res}")
            return result
        except qianfan.errors.QianfanError as e:
            raise ModelRunError(f"qianfan call failed: {e.__class__.__name__}({e})") from e
