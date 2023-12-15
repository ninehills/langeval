try:
    import openai
except ImportError as e:
    raise ValueError("Could not import openai python package. Please install it with `pip install openai`.") from e

import logging
from typing import Any, List

from langeval.models.exception import ModelRunError
from langeval.models.types import Message

logger = logging.getLogger(__name__)


class OpenAI:
    def __init__(self, model: str, chat: bool = True):
        self.model = model
        self.client = openai.Client()
        self.chat = chat

    def call(self, prompt: str, messages: List[Message], timeout: int, **kwargs: Any) -> str:
        try:
            if self.chat and prompt:
                # When chat model use prompt, then convert it to messages
                messages = [Message(role="user", content=prompt)]
            if messages:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": message.role, "content": message.content} for message in messages], # type: ignore
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
