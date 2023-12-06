import logging
from typing import Any, List
from string import Formatter

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
    # 支持动态生成 db_url，需要在db_url 中有 { db_name } 字段
    db_name_key: str = "db_id"

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
            # 判断 db_url 是否需要动态生成
            vars = get_template_variables(self.db_url)
            if "db_name" in vars:
                db_name = kwargs[self.db_name_key]
                db_url = self.db_url.format(db_name=db_name)
            else:
                db_url = self.db_url
            exact_match, correct = compare_query_results(
                query_gold=golden_sql,
                query_gen=sql,
                db_url=db_url,
                question=question,
                timeout=timeout,
            )
            ret["exact_match"] = int(exact_match)
            ret["correct"] = int(correct)
            ret["error_msg"] = ""
        except Exception as e:
            ret["error_msg"] = f"QUERY EXECUTION ERROR: {e}"
        return ret


def get_template_variables(template: str) -> List[str]:
    """Get the variables from the template.

    Args:
        template: The template string.
        template_format: The template format. Should be one of "f-string" or "jinja2".

    Returns:
        The variables from the template.

    Raises:
        ValueError: If the template format is not supported.
    """
    input_variables = {
        v for _, v, _, _ in Formatter().parse(template) if v is not None
    }


    return sorted(input_variables)
