import logging
from typing import Any

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

from .sqleval import compare_query_results  # noqa: TID252

logger = logging.getLogger(__name__)


class SQLEvaluator(pc.BaseModel):
    """SQL Evaluator

    from: <https://defog.ai/blog/open-sourcing-sqleval/>

    Output format:
    {
        "exact_match": 0/1,
        "correct": 0/1,
        "error_msg": "QUERY EXECUTION ERROR: ..."
    }
    """
    question_key: str
    sql_key: str
    # golden sql, 支持用 {a.c, a.b} 表示兼容 a.c 或 a.b
    golden_sql_key: str
    # sqlalchemy url, eg:
    # - sqlite:///tmp/test.db
    # - mysql://user:pass@localhost:port/dbname
    # - postgresql://user:pass@localhost:port/dbname
    db_url: str

    def call(self, kwargs: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
        """Evaluate call"""
        question = kwargs[self.question_key]
        sql = kwargs[self.sql_key]
        golden_sql = kwargs[self.golden_sql_key]

        ret = {
            "exact_match": 0,
            "correct": 0,
            "error_msg": "",
        }
        try:
            exact_match, correct = compare_query_results(
                query_gold=golden_sql,
                query_gen=sql,
                db_url=self.db_url,
                question=question,
                timeout=timeout,
            )
            ret["exact_match"] = int(exact_match)
            ret["correct"] = int(correct)
            ret["error_msg"] = ""
        except Exception as e:
            ret["error_msg"] = f"QUERY EXECUTION ERROR: {e}"
        return ret
