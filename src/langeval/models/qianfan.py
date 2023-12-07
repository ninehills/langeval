try:
    import qianfan
    import qianfan.errors
except ImportError as e:
    raise ValueError(
        "Could not import qianfan python package. Please install it with `pip install qianfan`."
    ) from e

from typing import Any

from langeval.models.exception import ModelRunError


class Qianfan:
    def __init__(self, model: str):
        self.model = model
        completion_models = set(qianfan.Completion._supported_models().keys())
        chat_models = set(qianfan.ChatCompletion._supported_models().keys())

        if model in (completion_models - chat_models):
            self.client = qianfan.Completion(model=model)
        elif model in chat_models:
            self.client = qianfan.ChatCompletion(model=model)
        else:
            self.client = qianfan.ChatCompletion(endpoint=model)

    def call(self, prompt: str, messages: list, timeout: int, **kwargs: Any) -> str:
        try:
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
