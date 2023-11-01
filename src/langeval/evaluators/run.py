import logging
from typing import Any

import jinja2

from langeval.evaluators.evaluator import EmbeddingCosSim, Evaluator, LLMGrade, PythonCode
from langeval.evaluators.exception import EvalRunError
from langeval.models import Embedding, ModelRunError
from langeval.providers.output_parser import OutputParserError, SimpleJsonOutputParser

logger = logging.getLogger(__name__)


def eval_python_code(evaluator: Evaluator, kwargs: dict[str, Any], timeout) -> dict[str, Any]:
    """Do python code eval"""
    if evaluator.settings is None or type(evaluator.settings) != PythonCode:
        msg = "PYTHON_CODE not specified"
        raise EvalRunError(msg)
    try:
        # Write code to temp file, then run it
        import json
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=True) as f:
            f.write(evaluator.settings.code)
            f.flush()
            result = subprocess.run(
                ["python3", f.name],  # noqa: S603, S607
                input=json.dumps(kwargs),
                stdout=subprocess.PIPE,
                timeout=timeout,
                check=True,
                encoding="utf-8",
            )
            return json.loads(result.stdout)
    except Exception as e:
        logger.warn(f"eval python code failed: {e}", exc_info=True)
        raise EvalRunError(f"eval python code failed: {e}") from e


def eval_embedding_cos_sim(evaluator: Evaluator, kwargs: dict[str, Any], timeout) -> dict[str, Any]:
    """Embedding Cosine Similarity"""
    if evaluator.settings is None or type(evaluator.settings) != EmbeddingCosSim:
        raise EvalRunError("EMBEDDING_COS_SIM not specified")
    if len(evaluator.input_keys) == 0 or len(evaluator.output_keys) == 0:
        raise EvalRunError("EMBEDDING_COS_SIM input/output keys not specified")
    try:
        model = evaluator.settings.embedding
        key1 = evaluator.input_keys[0]
        key2 = evaluator.output_keys[0]
        embeddings = model.embedding([kwargs[key1], kwargs[key2]], timeout=timeout)
        cos_sim = Embedding.cosine_similarity(embeddings[0], embeddings[1])
        is_similar = cos_sim >= evaluator.settings.cos_sim_threshold
        return {
            "cos_sim": cos_sim,
            "is_similar": int(is_similar),
        }
    except ModelRunError as e:
        raise EvalRunError(f"eval embedding cos_sim call failed: {e}") from e
    except Exception as e:
        logger.error(f"eval embedding cos_sim failed: {e}", exc_info=True)
        raise EvalRunError(f"eval embedding cos_sim failed: {e}") from e


def eval_llm_grade(evaluator: Evaluator, kwargs: dict[str, Any], timeout, default_llm) -> dict[str, Any]:
    """Use LLM as Judge for Grade"""
    if evaluator.settings is None or type(evaluator.settings) != LLMGrade:
        raise EvalRunError("LLM_GRADE not specified")
    if evaluator.settings.llm:
        llm = evaluator.settings.llm
    else:
        llm = default_llm
    if not llm:
        raise EvalRunError("LLM not specified")
    prompt = jinja2.Template(evaluator.settings.prompt).render(**kwargs)
    try:
        text = llm.completion(prompt, timeout=timeout)
    except ModelRunError as e:
        raise EvalRunError(f"eval completion failed: {e}") from e
    try:
        eval_result = {}
        result = SimpleJsonOutputParser().parse(text)
        for k in evaluator.eval_keys:
            if k not in result:
                raise EvalRunError(f"eval completion result missing key: {k}")
            eval_result[k] = result[k]
        return eval_result
    except OutputParserError as e:
        raise EvalRunError(f"eval completion output parser failed: {e}, src: {text}") from e
    except Exception as e:
        raise EvalRunError(f"eval completion parser failed: {e}, src: {text}") from e
