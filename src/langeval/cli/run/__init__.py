"""Run evaluation task."""
import os
import uuid
from datetime import datetime

import click

from langeval.cli.application import Application
from langeval.cli.constant import TaskOutputVars
from langeval.cli.run.display import save_task_result, show_task_result
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
@click.option(
    "--interactive", "-i", "interactive", is_flag=True, help="Interactively choose details about the evaluation"
)
@click.option("--web", "web", is_flag=True, help="Display the web UI for the evaluation.")
@click.option("--sample", "-s", "sample", type=int, help="Sample size for the evaluation.")
@click.option("--sample_seed", "-ss", "sample_seed", type=int, help="Sample seed for the evaluation. Default: 42", default=42)
@click.pass_obj
def run(app: Application, task_file, output, interactive, web, sample, sample_seed):
    """Run evaluation task.

    TASK_FILE: The evaluation task yaml file.
    """
    # 1. Load task
    task_file_content = task_file.read()
    task = EvalTask.from_yaml(task_file_content, dataset_dir=os.path.dirname(task_file.name))
    task_id = f"{datetime.now().strftime('%y%m%d%H%M')}-{uuid.uuid4().hex[:4]}"
    app.display_info(f">>> Loaded task from {task_file.name} successfully, task_id: {task_id}")

    # 2. Create output dir
    if not output:
        output = f"output/{task_id}"
    app.display_info(f"Output dir: {output}")
    if not os.path.exists(output):
        if interactive:
            if not app.confirm(f"Create output dir: {output}?"):
                app.abort(f"Output dir {output} not exists.")
        os.makedirs(output)
        app.display_info(f"Output dir created: {output}")

    # 3. Copy task file & input dataset to output dir
    with open(os.path.join(output, TaskOutputVars.TaskMeta), "w") as f:
        f.write(task_file_content)
    if task.input_dataset_name and task.input_dataset_binary:
        with open(os.path.join(output, task.input_dataset_name), "wb") as f:
            f.write(task.input_dataset_binary)

    # 4. Run task
    log_file = os.path.join(output, TaskOutputVars.TaskLog)
    log_file_handler = open(log_file, "w")

    def log_callback(_, log):
        app.display_info(log)
        log_file_handler.write(log)
        log_file_handler.flush()

    status_file = os.path.join(output, TaskOutputVars.TaskStatus)
    output_file = os.path.join(output, TaskOutputVars.TaskOutput)
    output_file_handler = open(output_file, "w")

    def progress_callback(_, progress, results):
        app.display_info(f"Progress: {progress}")
        jsonl = "".join([r.to_jsonl() for r in results])
        output_file_handler.write(jsonl)
        output_file_handler.flush()

        with open(status_file, "w") as f:
            f.write(runner.status_json())

    def status_callback(_, status):
        app.display_info(f"Status: {status}")
        with open(status_file, "w") as f:
            f.write(runner.status_json())

    runner = TaskRunner(
        task_id,
        task,
        sample=sample,
        sample_seed=sample_seed,
        status_callback=status_callback,
        log_callback=log_callback,
        progress_callback=progress_callback,
    )

    app.display_waiting(f">>> Start running task {task_id}")
    if interactive:
        if not app.confirm("Submit task ?"):
            app.abort("Task not submitted.")

    runner.start()
    app.display_waiting(f">>> Task {task_id} running...")
    runner.join()

    # 5. Task finish
    if runner.status == TaskRunnerStatus.FINISHED:
        app.display_success(f">>> Task {task_id} finish: {runner.status}.")
    else:
        app.display_error(f">>> Task {task_id} finish: {runner.status}.")

    log_file_handler.close()
    output_file_handler.close()

    with open(os.path.join(output, TaskOutputVars.TaskResult), "w") as f:
        for result in runner.results:
            f.write(result.to_jsonl())

    # 6. Show result
    running_stats, eval_stats = runner.statistic()
    app.display_info(f"Save task result to {output}")
    save_task_result(
        os.path.join(output, TaskOutputVars.TaskStastics),
        running_stats, eval_stats)

    show_task_result(app, runner, output)
