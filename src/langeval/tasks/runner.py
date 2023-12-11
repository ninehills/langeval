import enum
import json
import logging
import random
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

from langeval.models.llms import LLM
from langeval.tasks.task import EvalTask, Result

logger = logging.getLogger(__name__)


class TaskRunnerStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"

class Progress(pc.BaseModel):
    total: int = 0
    finished: int = 0
    failed: int = 0

class TaskProgress(pc.BaseModel):
    run: Progress
    evals: dict[str, Progress]


class TaskRunner:
    """
    Run task in background thread
    """

    def __init__(
        self,
        uuid: str,
        task: EvalTask,
        sample: int = 0,
        sample_seed: int = 42,
        default_eval_llm: Optional[LLM] = None,
        status_callback = None,
        log_callback = None,
        progress_callback = None,
    ) -> None:
        self.task = task
        self.uuid = uuid
        self.sample = sample
        self.sample_seed = sample_seed
        self.default_eval_llm = default_eval_llm

        # callback for status, log, progress updated.
        self.status_callback = status_callback
        self.log_callback = log_callback
        self.progress_callback = progress_callback

        # status with lock
        self._status_lock = threading.Lock()
        self._status = TaskRunnerStatus.PENDING

        self.thread = None
        self.finished_time = None
        self.cancel_event = threading.Event()

        self.progress: TaskProgress = TaskProgress(run=Progress(), evals={})
        self.results: list[Result] = []

    @property
    def status(self) -> TaskRunnerStatus:
        with self._status_lock:
            return self._status

    def status_json(self):
        return json.dumps(
            {
                "uuid": self.uuid,
                "status": self.status.value,
                "progress": self.progress.dict(),
                "finished_time": self.finished_time,
                "sample": self.sample,
                "sample_seed": self.sample_seed,
            }
        )

    def set_status(self, status: TaskRunnerStatus):
        with self._status_lock:
            self._status = status
        if self.status_callback:
            self.status_callback(self.uuid, status)

    def update_task_log(self, log: str):
        logger.info(f"[task-{self.uuid}]{log}")
        log_entry = f"[{datetime.utcnow().isoformat()}][task-{self.uuid}]{log}\n"
        if self.log_callback:
            self.log_callback(self.uuid, log_entry)

    def update_task_progress(self, progress: TaskProgress, results: list[Result]):
        self.progress = progress
        if self.progress_callback:
            self.progress_callback(self.uuid, progress, results)

    def start(self):
        """
        启动任务运行器
        """
        self.update_task_log("[runner.start] start task runner")
        self.set_status(TaskRunnerStatus.RUNNING)
        try:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except Exception as e:
            logger.error(f"[task-{self.uuid}][runner.start] failed to start task runner: {e}")
            self.update_task_log(f"[runner.start] failed to start task runner: {e}")
            self.set_status(TaskRunnerStatus.FAILED)

    def join(self, timeout=None):
        """
        等待任务运行结束
        """
        if self.thread:
            self.thread.join(timeout=timeout)

    def cancel(self):
        """
        取消任务
        """
        self.update_task_log("[runner.cancel] cancel task runner")
        self.set_status(TaskRunnerStatus.CANCELLED)
        self.cancel_event.set()

    def _run(self) -> None:
        """
        运行任务
        """
        try:
            self.update_task_log(f"[runner._run] start task run : {self.task.run_config}")

            data = self.task.split_dataset()
            data_lists = []
            for d in data:
                data_lists.extend([d] * self.task.run_config.rounds)

            self.update_task_log(
                f"[runner._run] task run: {len(data)} * {self.task.run_config.rounds} = {len(data_lists)}"
            )
            if self.sample and len(data_lists) > 0:
                self.update_task_log("[runner._run] task sample to 1 data.")
                data_lists = random.Random(self.sample_seed).sample(data_lists, self.sample)
            total = len(data_lists)
            if self.results:
                self.update_task_log(f"[runner._run] task resume from {len(self.results)} results.")
            else:
                self.results = [Result(inputs=d) for d in data_lists]
            if self.task.provider is not None:
                self.update_task_log("[runner._run] provider start run.")
                # get finished result
                new_results = []
                need_run = []
                for result in self.results:
                    if result.run.error or not result.run.outputs:
                        need_run.append(result)
                    else:
                        new_results.append(result)
                self.update_task_log(f"[runner._run] provider resume from {len(new_results)} results.")
                progress = TaskProgress(run=Progress(total=total, finished=len(new_results)), evals={})

                for result in self.task.run_provider(need_run, self.cancel_event):
                    if result.run.error:
                        progress.run.failed += 1
                    else:
                        progress.run.finished += 1
                    self.update_task_log(f"[runner._run] task progress {progress}, result: {result}")
                    self.update_task_progress(progress, [result])
                    new_results.append(result)

                    # Check if task be cancelled
                    if self.cancel_event.is_set():
                        logger.warn(f"[task-{self.uuid}][runner._run] task be cancelled")
                        self.set_status(TaskRunnerStatus.CANCELLED)
                        self.update_task_log("[runner._run] task cancelled")
                        return
                self.results = new_results
                logger.info(f"[task-{self.uuid}][runner._run] end task run")
                if progress.run.failed == progress.run.total:
                    self.finished_time = time.time()
                    self.set_status(TaskRunnerStatus.FAILED)
                    self.update_task_log("[runner._run] task failed because all run failed")
                    return

            progress = self.progress
            for evaluator in self.task.evaluators:
                self.update_task_log(f"[runner._run] evaluator {evaluator.name} start run.")
                # get finished result
                new_results = []
                need_run = []
                for result in self.results:
                    eval_result = result.evals.get(evaluator.name)
                    if not eval_result or eval_result.error or not eval_result.outputs:
                        need_run.append(result)
                    else:
                        new_results.append(result)
                self.update_task_log(f"[runner._run] evaluator {evaluator.name} resume from {len(new_results)} results.")
                progress.evals[evaluator.name] = Progress(total=total, finished=len(new_results))
                for result in self.task.run_eval(need_run, self.cancel_event,
                                                 default_eval_llm=self.default_eval_llm):
                    if result.evals[evaluator.name].error:
                        progress.evals[evaluator.name].failed += 1
                    else:
                        progress.evals[evaluator.name].finished += 1
                    self.update_task_log(
                        f"[runner._run] task eval {evaluator.name} progress "
                        "{progress.evals[evaluator.name]}, result: {result}")
                    self.update_task_progress(progress, [result])
                    new_results.append(result)

                    # Check if task be cancelled
                    if self.cancel_event.is_set():
                        logger.warn(f"[task-{self.uuid}][runner._run] task be cancelled")
                        self.set_status(TaskRunnerStatus.CANCELLED)
                        self.update_task_log("[runner._run] task cancelled")
                        return
                logger.info(f"[task-{self.uuid}][runner._run] end task eval")
                self.results = new_results

        except Exception as e:
            logger.error(f"[task-{self.uuid}][runner._run] failed to run task : {e}", exc_info=True)
            self.set_status(TaskRunnerStatus.FAILED)
            self.update_task_log(f"[runner._run] failed to run task : {e}")

    def statistic(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.DataFrame([i.dict() for i in self.results])
        total_count = len(df)
        run_success_rate = 0
        run_success_count = 0
        run_average_time = 0
        eval_success_rate = 0
        eval_average_time = 0
        if "run" in df.columns:
            run_success_rate = df["run"].apply(lambda x: x["error"] == "").mean()
            run_success_count = df["run"].apply(lambda x: x["error"] == "").sum()
            run_average_time = df["run"].apply(lambda x: x["elapsed_secs"]).mean()
        if "evals" in df.columns:
            eval_success_rate = df["evals"].apply(lambda x: all(e["error"] == "" for e in x.values())).mean()
            eval_average_time = df["evals"].apply(lambda x: sum(e["elapsed_secs"] for e in x.values())).mean()
        running_stats = pd.DataFrame(
            [
                {
                    "Total count": f"{total_count}",
                    "Run success rate": f"{run_success_rate:.2%}",
                    "Run success count": f"{run_success_count}",
                    "Run average secs": f"{run_average_time:.2f}",
                    "Eval success rate": f"{eval_success_rate:.2%}",
                    "Eval average secs": f"{eval_average_time:.2f}",
                }
            ]
        )
        eval_stats = pd.DataFrame()

        # "evals": {"exact_match": {
        #   "error": "", "outputs": {"exact_match": 1.0}, "elapsed_secs": 5.728999894927256e-06}}
        def flatten_outputs(data_row):
            """
            从嵌套字典中提取并展平 'outputs' 键下的内容。
            :param data_row: 包含嵌套字典的数据行。
            :return: 展平后的 'outputs' 字典。
            """
            flattened = {}
            for key, value in data_row.items():
                if "outputs" in value:
                    for output_key, output_value in value["outputs"].items():
                        flattened_key = f"{key}.outputs.{output_key}"
                        flattened[flattened_key] = output_value
            return flattened
        if "evals" in df.columns:
            flattened_evals = df["evals"].apply(flatten_outputs)
            flattened_df = pd.DataFrame(flattened_evals.tolist())
            flattened_df.fillna(0.0, inplace=True)

            if not flattened_df.empty:
                eval_stats = pd.DataFrame(flattened_df).describe().T
        return running_stats, eval_stats
