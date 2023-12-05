import json
import logging
import os
import subprocess
from typing import Any

import jinja2

from langeval.models import ModelRunError
from langeval.providers.exception import ProviderRunError
from langeval.providers.provider import ExecSettings, LLMSettings, Provider

logger = logging.getLogger(__name__)


def call_completion(conf: Provider, inputs: dict[str, Any], timeout: int) -> dict[str, str]:
    if conf.settings is None or not isinstance(conf.settings, LLMSettings):
        raise ProviderRunError(f"call_completion invalid provider config: {conf}")

    prompt = jinja2.Template(conf.settings.prompt).render(**inputs)

    try:
        text = conf.settings.llm.completion(prompt, timeout=timeout)
    except ModelRunError as e:
        raise ProviderRunError(f"call_completion failed: {e}") from e
    except Exception as e:
        raise ProviderRunError(f"call_completion failed: {e}") from e

    return conf.output_parser.parse(text)


def call_chat_completion(conf: Provider, inputs: dict[str, Any], timeout: int):
    if conf.settings is None or not isinstance(conf.settings, LLMSettings):
        raise ProviderRunError(f"call_completion invalid provider config: {conf}")
    messages = json.loads(conf.settings.prompt)
    for message in messages:
        message["content"] = jinja2.Template(message["content"]).render(**inputs)


    try:
        text = conf.settings.llm.chat_completion(messages, timeout=timeout)
    except ModelRunError as e:
        raise ProviderRunError(f"call_chat_completion failed: {e}") from e
    except Exception as e:
        raise ProviderRunError(f"call_chat_completion failed: {e}") from e

    return conf.output_parser.parse(text)


def batch_call_exec(conf: Provider, inputs_list: list[dict[str, Any]], timeout: int) -> list[dict[str, Any]]:
    if conf.settings is None or not isinstance(conf.settings, ExecSettings):
        raise ProviderRunError(f"call_exec invalid provider config: {conf}")
    command = conf.settings.command
    kwargs = conf.settings.kwargs or {}
    # Copy progress env.
    env = os.environ.copy()
    if kwargs.get("env"):
        env.update(kwargs["env"])
    cwd = kwargs.get("cwd") or None
    exec_timeout = int(kwargs.get("timeout", 300))  # type: ignore
    timeout = min(timeout, exec_timeout)
    input_data = json.dumps(inputs_list, ensure_ascii=False)

    try:
        cp = subprocess.run(
            command,
            shell=True,  # noqa: S602
            check=True,
            encoding="utf-8",
            env=env,
            cwd=cwd,
            input=input_data,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise ProviderRunError(f"call_exec failed: {e}") from e
    except Exception as e:
        raise ProviderRunError(f"call_exec failed: {e}") from e

    try:
        # list for string
        texts = json.loads(cp.stdout)
    except json.JSONDecodeError as e:
        raise ProviderRunError(f"call_exec output parser failed: {e}") from e

    return [conf.output_parser.parse(text) for text in texts]
