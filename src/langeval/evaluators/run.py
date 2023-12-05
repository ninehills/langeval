import logging
from typing import Any

import jinja2

from langeval.evaluators.evaluator import EmbeddingCosSim, Evaluator, LLMGrade, PythonCode
from langeval.evaluators.exception import EvalRunError
from langeval.models import Embedding
from langeval.providers.output_parser import SimpleJsonOutputParser

logger = logging.getLogger(__name__)


def eval_python_code(evaluator: Evaluator, kwargs_list: list[dict[str, Any]], timeout) -> list[dict[str, Any]]:
    """Do python code eval"""
    if evaluator.settings is None or not isinstance(evaluator.settings, PythonCode):
        msg = "PYTHON_CODE not specified"
        raise EvalRunError(msg)

    # Write code to temp file, then run it
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=True) as f:
        f.write(evaluator.settings.code)
        f.flush()
        result = subprocess.run(
            ["python3", f.name],  # noqa: S603, S607
            input=json.dumps(kwargs_list),
            stdout=subprocess.PIPE,
            timeout=timeout,
            check=True,
            encoding="utf-8",
        )
        return json.loads(result.stdout)

def eval_embedding_cos_sim(evaluator: Evaluator, kwargs: dict[str, Any], timeout) -> dict[str, Any]:
    """Embedding Cosine Similarity"""
    if evaluator.settings is None or not isinstance(evaluator.settings, EmbeddingCosSim):
        raise EvalRunError("EMBEDDING_COS_SIM not specified")
    if len(evaluator.settings.pairs_keys) != 2: # noqa: PLR2004
        raise EvalRunError("EMBEDDING_COS_SIM input/output keys not specified")

    model = evaluator.settings.embedding
    key1, key2 = evaluator.settings.pairs_keys
    embeddings = model.embedding([kwargs[key1], kwargs[key2]], timeout=timeout)
    cos_sim = Embedding.cosine_similarity(embeddings[0], embeddings[1])
    is_similar = cos_sim >= evaluator.settings.cos_sim_threshold
    return {
        "cos_sim": cos_sim,
        "is_similar": int(is_similar),
    }

def eval_llm_grade(evaluator: Evaluator, kwargs: dict[str, Any], timeout, default_llm) -> dict[str, Any]:
    """Use LLM as Judge for Grade"""
    if evaluator.settings is None or not isinstance(evaluator.settings, LLMGrade):
        raise EvalRunError("LLM_GRADE not specified")
    if evaluator.settings.llm:
        llm = evaluator.settings.llm
    else:
        llm = default_llm
    if not llm:
        raise EvalRunError("LLM not specified")
    prompt = jinja2.Template(evaluator.settings.prompt).render(**kwargs)
    text = llm.completion(prompt, timeout=timeout)
    eval_result = {}
    result = SimpleJsonOutputParser().parse(text)
    for k in evaluator.settings.eval_keys:
        if k not in result:
            raise EvalRunError(f"eval completion result missing key: {k}")
        eval_result[k] = result[k]
    return eval_result
