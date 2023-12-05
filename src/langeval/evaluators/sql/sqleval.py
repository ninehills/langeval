# this file contains all of the helper functions used for evaluations
# from: https://github.com/defog-ai/sql-eval/blob/main/eval/eval.py
# Licensed under the Apache-2.0 License

import itertools
import re
from venv import logger

import pandas as pd
from func_timeout import func_timeout
from pandas.testing import assert_frame_equal, assert_series_equal
from sqlalchemy import create_engine

# like_pattern = r"LIKE\s+'[^']*'"
LIKE_PATTERN = r"LIKE[\s\S]*'"


def normalize_table(
    df: pd.DataFrame, query_category: str, question: str
) -> pd.DataFrame:
    """
    Normalizes a dataframe by:
    1. removing all duplicate rows
    2. sorting columns in alphabetical order
    3. sorting rows using values from first column to last (if query_category is not 'order_by' and
        question does not ask for ordering)
    4. resetting index
    """
    # remove duplicate rows, if any
    df = df.drop_duplicates()

    # sort columns in alphabetical order
    sorted_df = df.reindex(sorted(df.columns), axis=1)

    # check if query_category is 'order_by' and if question asks for ordering
    has_order_by = False
    pattern = re.compile(r"(order|sort|arrange)", re.IGNORECASE)
    in_question = re.search(pattern, question.lower())  # true if contains
    if query_category == "order_by" or in_question:
        has_order_by = True
    if not has_order_by:
        # sort rows using values from first column to last
        sorted_df = sorted_df.sort_values(by=list(sorted_df.columns))
    # reset index
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df


# for escaping percent signs in regex matches
def escape_percent(match):
    # Extract the matched group
    group = match.group(0)
    # Replace '%' with '%%' within the matched group
    escaped_group = group.replace("%", "%%")
    # Return the escaped group
    return escaped_group


# find start and end index of { } in a string. return (start, end) if found, else return (-1, -1)
def find_bracket_indices(s: str, start_index: int = 0) -> "tuple[int, int]":
    start = s.find("{", start_index)
    end = s.find("}", start + 1)
    if start == -1 or end == -1:
        return (-1, -1)
    return (start, end)


# extrapolate all possible queries from a query with { } in it
def get_all_minimal_queries(query: str) -> "list[str]":
    """
    extrapolate all possible queries
    - split by semicolon. this is to accommodate queries where joins to other tables are also acceptable.
    - expand all column permutations if there are braces { } in it. eg:
    ```sql
        SELECT {user.id, user.name} FROM user;
    ```
    Would be expanded to:
    ```sql
        SELECT user.id FROM user;
        SELECT user.name FROM user;
        SELECT user.id, user.name FROM user;
    ```
    """
    queries = query.split(";")
    result_queries = []
    for q in queries:
        query = q.strip()
        if query == "":
            continue
        start, end = find_bracket_indices(query, 0)
        if (start, end) == (-1, -1):
            result_queries.append(query)
            continue
        else:
            # get all possible column subsets
            column_options = query[start + 1 : end].split(",")
            column_combinations = list(
                itertools.chain.from_iterable(
                    itertools.combinations(column_options, r)
                    for r in range(1, len(column_options) + 1)
                )
            )
            for column_tuple in column_combinations:
                left = query[:start]
                column_str = ", ".join(column_tuple)
                right = query[end + 1 :]
                # change group by size dynamically if necessary
                if right.find("GROUP BY {}"):
                    right = right.replace("GROUP BY {}", f"GROUP BY {column_str}")
                result_queries.append(left + column_str + right)
    return result_queries


def query_db(
    query: str, db_url: str, timeout: float = 10.0
) -> pd.DataFrame:
    """
    Runs query on postgres db and returns results as a dataframe.
    This assumes that you have the evaluation database running locally.
    If you don't, you can following the instructions in the README (Restoring to Postgres) to set it up.

    timeout: time in seconds to wait for query to finish before timing out
    """
    try:
        engine = create_engine(db_url)
        escaped_query = re.sub(
            LIKE_PATTERN, escape_percent, query, flags=re.IGNORECASE
        )  # ignore case of LIKE
        results_df = func_timeout(
            timeout, pd.read_sql_query, args=(escaped_query, engine)
        )
        engine.dispose()  # type: ignore
        return results_df # type: ignore
    except Exception as e:
        if engine: # type: ignore
            engine.dispose()  # type: ignore
        raise e


def compare_df(
    df1: pd.DataFrame, df2: pd.DataFrame, query_category: str, question: str
) -> bool:
    """
    Compares two dataframes and returns True if they are the same, else False.
    """
    # drop duplicates to ensure equivalence
    if df1.shape == df2.shape and (df1.values == df2.values).all():
        return True

    df1 = normalize_table(df1, query_category, question)
    df2 = normalize_table(df2, query_category, question)

    if df1.shape == df2.shape and (df1.values == df2.values).all():
        return True
    else:
        return False


def subset_df(
    df_sub: pd.DataFrame,
    df_super: pd.DataFrame,
    query_category: str,
    question: str,
    verbose: bool = False,
) -> bool:
    """
    Checks if df_sub is a subset of df_super
    """
    if df_sub.empty:
        return False  # handle cases for empty dataframes

    # make a copy of df_super so we don't modify the original while keeping track of matches
    df_super_temp = df_super.copy(deep=True)
    matched_columns = []
    for col_sub_name in df_sub.columns:
        col_match = False
        for col_super_name in df_super_temp.columns:
            col_sub = df_sub[col_sub_name].sort_values().reset_index(drop=True)
            col_super = (
                df_super_temp[col_super_name].sort_values().reset_index(drop=True)
            )
            try:
                assert_series_equal(
                    col_sub, col_super, check_dtype=False, check_names=False
                )
                col_match = True
                matched_columns.append(col_super_name)
                # remove col_super_name to prevent us from matching it again
                df_super_temp = df_super_temp.drop(columns=[col_super_name])
                break
            except AssertionError:
                continue
        if col_match is False:
            if verbose:
                logger.warning(f"no match for {col_sub_name}")
            return False
    df_sub_normalized = normalize_table(df_sub, query_category, question)

    # get matched columns from df_super, and rename them with columns from df_sub, then normalize
    df_super_matched = df_super[matched_columns].rename(
        columns=dict(zip(matched_columns, df_sub.columns))
    )
    df_super_matched = normalize_table(df_super_matched, query_category, question)

    try:
        assert_frame_equal(df_sub_normalized, df_super_matched, check_dtype=False)
        return True
    except AssertionError:
        return False


def compare_query_results(
    query_gold: str,
    query_gen: str,
    db_url: str,
    question: str,
    timeout: float = 10.0,
) -> "tuple[bool, bool]":
    """
    Compares the results of two queries and returns a tuple of booleans, where the first element is
    whether the queries produce exactly the same result, and the second element is whether the
    result of the gold query is a subset of the result of the generated query (still correct).
    We bubble up exceptions (mostly from query_postgres_db) to be handled in the runner.
    """
    # check if query contains "order by"
    query_category = "order_by" if "order by" in query_gold.lower() else ""
    queries_gold = get_all_minimal_queries(query_gold)
    results_gen = query_db(query_gen, db_url, timeout)
    correct = False
    for q in queries_gold:
        results_gold = query_db(q, db_url, timeout)
        if compare_df(results_gold, results_gen, query_category, question):
            return (True, True)
        elif subset_df(results_gold, results_gen, query_category, question):
            correct = True
    return (False, correct)
