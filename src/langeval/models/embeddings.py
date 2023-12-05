import logging

import numpy as np
from numpy.linalg import norm

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

from langeval.models.exception import ModelRunError

logger = logging.getLogger(__name__)


class Embedding(pc.BaseModel):
    provider: str
    model: str
    # Model parameters, e.g. Qianfan has ak, sk
    kwargs: dict = {}

    @pc.validator("provider")
    def provider_must_be_valid(cls, v):  # noqa: N805
        if v not in ["qianfan", "openai"]:
            raise ValueError(f"Invalid provider: {v}")
        return v

    def embedding(self, texts: list[str], timeout: int = 10) -> list[list[float]]:
        """Generate embeddings for texts"""
        if self.provider == "qianfan":
            # Cut for qianfan 384 tokens limit
            texts = [text[:384] for text in texts]
            import qianfan
            import qianfan.errors

            try:
                client = qianfan.Embedding(**self.kwargs)
                res = client.do(texts, request_timeout=float(timeout))
                logger.debug(f"qianfan embedding: {texts}")
                if res.code != 200:  # type: ignore  # noqa: PLR2004
                    raise ModelRunError(f"qianfan embedding failed: {res}")
                result = res.body  # type: ignore
                if not result:
                    raise ModelRunError(f"qianfan embedding failed: {res}")
                # type: ignore
                return [i.get("embedding", []) for i in result["data"]]
            except qianfan.errors.QianfanError as e:
                raise ModelRunError(f"qianfan embedding failed: {e.__class__.__name__}({e})") from e
            except Exception as e:
                logger.error(f"qianfan embedding failed: {e}", exc_info=True)
                raise ModelRunError(f"qianfan embedding failed: {e}") from e
        elif self.provider == "openai":
            try:
                import openai
            except ImportError as e:
                raise ValueError(
                    "Could not import openai python package. Please install it with `pip install openai`."
                ) from e
            try:
                response = openai.embeddings.create(
                    model=self.model, input=texts, encoding_format="float", timeout=timeout, **self.kwargs
                )
                logger.debug(f"openai embedding: {texts}")
                return [i.get("embedding", []) for i in response["data"]]  # type: ignore
            except Exception as e:
                raise ModelRunError(f"openai call failed: {e.__class__.__name__}({e})") from e
        else:
            raise NotImplementedError()

    @staticmethod
    def cosine_similarity(vector1: list[float], vector2: list[float]) -> float:
        """Compute cosine similarity between two vectors"""
        return np.dot(vector1, vector2) / (norm(vector1) * norm(vector2))
