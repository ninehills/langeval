"""Re-run the evaluation task."""
import json
import os
import shutil
import uuid
from datetime import datetime

import click

from langeval.cli.application import Application
from langeval.cli.constant import TaskOutputVars
from langeval.cli.run.run import run_task
from langeval.tasks import EvalTask, Result


@click.command(short_help="Re-run evaluation task.")
@click.argument("task_dir", required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output",
    help="Output directory for the evaluation files and results",
    type=click.Path(exists=False),
)
@click.pass_obj
def rerun(app: Application, task_dir, output):
    """Re-run evaluation task.

    TASK_DIR: The directory of the evaluation task.
    """
    # 1. 创建 output dir
    task_id = f"{datetime.now().strftime('%y%m%d%H%M')}-{uuid.uuid4().hex[:4]}"
    if not output:
        output = f"output/{task_id}"
    app.display_info(f"Output dir: {output}")
    if not os.path.exists(output):
        os.makedirs(output)
        app.display_info(f"Output dir created: {output}")
    else:
        app.abort(f"Output dir {output} exists, exit.")

    # 2. 复制 task_dir 到新的目录
    shutil.copytree(task_dir, output, dirs_exist_ok=True)

    # 3. Load task
    task_file = os.path.join(output, TaskOutputVars.TaskMeta)
    status_file = os.path.join(output, TaskOutputVars.TaskStatus)

    with open(task_file) as f:
        task_file_content = f.read()
    task = EvalTask.from_yaml(task_file_content, dataset_dir=output)
    with open(status_file) as f:
        # {"uuid": "2311021530-5c69", "status": "FINISHED", "progress": "1/0/1", "finished_time": 1698910215.125846}
        status = json.loads(f.read())
    with open(os.path.join(output, TaskOutputVars.TaskResult)) as f:
        results = [Result.from_json(line) for line in f.readlines()]

    # 4. Run task
    run_task(app, output, task_id, task,
             sample=status["sample"], sample_seed=status["sample_seed"],
             results=results)
