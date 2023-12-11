import json
import os

import pandas as pd

from langeval.cli.application import Application
from langeval.cli.constant import TaskOutputVars
from langeval.tasks import Result, TaskRunner


def save_task_merged_result(file: str, results: list[Result]):
    with open(file, "w") as f:
        for result in results:
            data = result.inputs
            data.update(result.run.outputs)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

def save_task_result(file: str, running_stats, eval_stats):
    result = {
        # process NaN value.
        "running_stats": json.loads(running_stats.to_json(force_ascii=False)),
        "eval_stats": json.loads(eval_stats.to_json(force_ascii=False)),
    }
    with open(file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

def show_task_result(app: Application, runner: TaskRunner, output_dir: str):
    result_file = os.path.join(output_dir, TaskOutputVars.TaskResult)
    # Display info
    app.display_header("Task Info")
    app.display_info(f"ID: {runner.uuid}")
    app.display_info(f"Status: {runner.status}")
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
