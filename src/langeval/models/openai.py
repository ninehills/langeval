try:
    import openai
except ImportError as e:
    raise ValueError("Could not import openai python package. Please install it with `pip install openai`.") from e

import logging
from typing import Any

from langeval.models.exception import ModelRunError

logger = logging.getLogger(__name__)


class OpenAI:
    def __init__(self, model: str):
        self.model = model
        self.client = openai.Client()

    def call(self, prompt: str, messages: list, timeout: int, **kwargs: Any) -> str:
        try:
            if not self.model.endswith("-instruct") and prompt:
                # When chat model use prompt, then convert it to messages
                messages = [{"role": "user", "content": prompt}]
            if messages:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=float(timeout),
                    **kwargs,
                )
                logger.debug(f"openai completion: {messages} -> {response}")
                return response.choices[0].message.content
            else:
                kwargs = kwargs.copy()
                if "max_tokens" not in kwargs:
                    # Default to 1024 tokens
                    kwargs["max_tokens"] = 1024
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    timeout=float(timeout),
                    **kwargs,
                )
                logger.debug(f"openai completion: {prompt} -> {response}")
                return response.choices[0].text
        except openai.OpenAIError as e:
            raise ModelRunError(f"openai call failed: {e.__class__.__name__}({e})") from e
