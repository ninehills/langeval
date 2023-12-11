"""Run evaluation task."""
import os
import uuid
from datetime import datetime

import click

from langeval.cli.application import Application
from langeval.cli.constant import TaskOutputVars
from langeval.cli.run.display import save_task_merged_result, save_task_result, show_task_result
from langeval.cli.run.run import run_task
from langeval.tasks import EvalTask, TaskRunner, TaskRunnerStatus


@click.command(short_help="Run evaluation task")
@click.argument("task_file", required=True, type=click.File("r"))
@click.option(
    "--output",
    "-o",
    "output",
    help="Output directory for the evaluation files and results",
    type=click.Path(exists=False),
)
@click.option("--sample", "-s", "sample", type=int, help="Sample size for the evaluation.")
@click.option("--sample_seed", "-ss", "sample_seed", type=int, help="Sample seed for the evaluation. Default: 42", default=42)
@click.pass_obj
def run(app: Application, task_file, output, sample, sample_seed):
    """Run evaluation task.

    TASK_FILE: The evaluation task yaml file.
    """
    # 1. Load task
    task_file_content = task_file.read()
    task = EvalTask.from_yaml(task_file_content)
    task_id = f"{datetime.now().strftime('%y%m%d%H%M')}-{uuid.uuid4().hex[:4]}"
    app.display_info(f">>> Loaded task from {task_file.name} successfully, task_id: {task_id}")

    # 2. Create output dir
    if not output:
        output = f"output/{task_id}"
    app.display_info(f"Output dir: {output}")
    if not os.path.exists(output):
        os.makedirs(output)
        app.display_info(f"Output dir created: {output}")
    else:
        app.abort(f"Output dir {output} exists, exit.")

    # 3. Copy task file & input dataset to output dir
    with open(os.path.join(output, TaskOutputVars.TaskMeta), "w") as f:
        f.write(task_file_content)
    if task.input_dataset_name and task.input_dataset_binary:
        input_dataset_filename = os.path.basename(task.input_dataset_name)
        with open(os.path.join(output, input_dataset_filename), "wb") as f:
            f.write(task.input_dataset_binary)
    # 4. Run task
    run_task(app, output, task_id, task, sample, sample_seed)
