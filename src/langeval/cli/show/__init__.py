"""Show evalution result in CLI or Web UI."""
import json
import os

import click

from langeval.cli.application import Application
from langeval.cli.constant import TaskOutputVars
from langeval.cli.run.display import show_task_result
from langeval.tasks import EvalTask, Result, TaskRunner


@click.command(short_help="Show evaluation result.")
@click.argument("task_dir", required=True, type=click.Path(exists=True))
@click.option("--web", "web", is_flag=True, help="Display the web UI for the evaluation.")
@click.pass_obj
def show(app: Application, task_dir, web):
    """Show evaluation result.

    TASK_DIR: The directory of the evaluation task.
    """
    task_file = os.path.join(task_dir, TaskOutputVars.TaskMeta)
    with open(task_file) as f:
        task_file_content = f.read()
    task = EvalTask.from_yaml(task_file_content, dataset_dir=task_dir)

    with open(os.path.join(task_dir, TaskOutputVars.TaskStatus)) as f:
        # {"uuid": "2311021530-5c69", "status": "FINISHED", "progress": "1/0/1", "finished_time": 1698910215.125846}
        status = json.loads(f.read())

    with open(os.path.join(task_dir, TaskOutputVars.TaskResult)) as f:
        results = [Result.from_json(line) for line in f.readlines()]

    runner = TaskRunner(status["uuid"], task)
    runner.set_status(status["status"])
    runner.progress = status["progress"]
    runner.finished_time = status["finished_time"]
    runner.results = results

    if not runner:
        app.abort(f"No task found in {task_dir}")
    show_task_result(app, runner, task_dir)
