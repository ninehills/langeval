import os
from typing import List, Optional

from langeval.cli.application import Application
from langeval.cli.constant import TaskOutputVars
from langeval.cli.run.display import save_task_merged_result, save_task_result, show_task_result
from langeval.tasks import EvalTask, Result, TaskRunner, TaskRunnerStatus


def run_task(app:Application, output: str, task_id: str, task: EvalTask,
             sample: int = 0, sample_seed: int = 42, results: Optional[List[Result]] = None):
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
    if results:
        app.display(">>> Load previous results.")
        runner.results = results

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

    save_task_merged_result(
        os.path.join(output, TaskOutputVars.TaskMerged),
        runner.results
    )

    show_task_result(app, runner, output)
