import json
import os

import pandas as pd

from langeval.cli.application import Application
from langeval.cli.constant import TaskOutputVars
from langeval.tasks import Result, TaskRunner


def show_task_result(app: Application, runner: TaskRunner, output_dir: str):
    result_file = os.path.join(output_dir, TaskOutputVars.TaskResult)
    # Display info
    app.display_header("Task Info")
    app.display_info(f"ID: {runner.uuid}")
    app.display_info(f"Status: {runner.status!s}")
    app.display_info(f"Progress: {runner.progress}")
    app.display_info(f"Output JSONL: {result_file}")

    # Display stastics
    app.display_header("Task Stastics")
    running_stats, eval_stats = runner.statistic()
    app.display_table(
        title="Run stats",
        columns=convert_running_stats_to_columns(running_stats),
        show_lines=True,
        force_ascii=True,
    )
    app.display_table(
        title="Eval stats",
        columns=convert_eval_stats_to_columns(eval_stats),
        show_lines=True,
        force_ascii=True,
    )

    # Display Result Sample
    app.display_header(f"Result Sample, see {result_file} for all results.")
    success_results = [r for r in runner.results if r.eval_error == "" and r.run_error == ""]
    if success_results:
        app.display_table(
            title="Success result sample",
            columns=convert_results_to_columns(success_results[:3]),
            show_lines=True,
            force_ascii=True,
        )

    failed_results = [r for r in runner.results if r.eval_error != "" or r.run_error != ""]
    if failed_results:
        app.display_table(
            title="Failed result sample",
            columns=convert_results_to_columns(failed_results[:3]),
            show_lines=True,
            force_ascii=True,
        )


def convert_results_to_columns(results) -> dict[str, dict[int, str]]:
    columns: dict[str, dict[int, str]] = {k: {} for k in Result.__annotations__.keys()}
    for i, r in enumerate(results):
        for k, v in r.__dict__.items():
            if isinstance(v, dict):
                v_copy = v.copy()
                for key in v:
                    if key.startswith("_"):
                        del v_copy[key]
                columns[k][i] = json.dumps(v_copy, indent=2, ensure_ascii=False)
            else:
                columns[k][i] = str(v)
    return columns


def convert_running_stats_to_columns(df: pd.DataFrame) -> dict[str, dict[int, str]]:
    columns: dict[str, dict[int, str]] = {k: {} for k in df.columns}
    index = 0
    for _, r in df.iterrows():
        for k, v in r.items():
            columns[str(k)][index] = str(v)
        index += 1
    return columns


def convert_eval_stats_to_columns(df: pd.DataFrame) -> dict[str, dict[int, str]]:
    columns: dict[str, dict[int, str]] = {"eval": {}}
    for k in df.columns:
        columns[k] = {}
    index = 0
    for i, r in df.iterrows():
        columns["eval"][index] = str(i)
        for k, v in r.items():
            columns[str(k)][index] = str(v)
        index += 1
    return columns
