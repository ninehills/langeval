import csv
import io
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
from typing import Any, List, Optional

import pandas as pd
import yaml

try:
    import pydantic.v1 as pc
except ImportError:
    import pydantic as pc

from langeval.evaluators import Evaluator
from langeval.models import LLM
from langeval.providers import Provider

logger = logging.getLogger(__name__)

class RunResult(pc.BaseModel):
    """Result for one data run"""
    error: str = ""
    outputs: dict[str, Any] = {}
    elapsed_secs: float = 0

class EvalResult(pc.BaseModel):
    """Result for one data eval"""
    error: str = ""
    outputs: dict[str, Any] = {}
    elapsed_secs: float = 0


class Result(pc.BaseModel):
    # uuid: str
    inputs: dict[str, Any]
    run: RunResult = RunResult()
    evals: dict[str, EvalResult] = {}

    def to_jsonl(self) -> str:
        return json.dumps(self.dict(), ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, json_str: str) -> "Result":
        return cls(**json.loads(json_str))


class TaskRunConfig(pc.BaseModel):
    parallelism: int = pc.Field(default=1, ge=1, le=30)
    timeout: int = pc.Field(default=30, ge=1, le=600)
    rounds: int = pc.Field(default=1, ge=1, le=10)
    batch_size: int = pc.Field(default=1, ge=1, le=10000)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.dict(exclude_unset=True), encoding="utf-8", allow_unicode=True).decode("utf-8")

    @staticmethod
    def from_yaml(yaml_str: str) -> "TaskRunConfig":
        try:
            return TaskRunConfig(**yaml.safe_load(yaml_str))
        except Exception as e:
            raise ValueError(f"Invalid yaml: {e}") from e


class EvalTask(pc.BaseModel):
    # config
    provider: Optional[Provider] = None
    input_dataset_binary: Optional[bytes] = None
    # Only jsonl, csv supported
    input_dataset_name: str = pc.Field(..., min_length=1, max_length=255)
    # evaluator
    evaluators: List[Evaluator]
    # Run config
    run_config: TaskRunConfig

    class Config:
        validate_assignment = True

    @pc.validator("input_dataset_name")
    def input_dataset_name_must_be_valid(cls, v):  # noqa: N805
        if not v.endswith(".csv") and not v.endswith(".jsonl"):
            raise ValueError(f"Invalid input_dataset_name: {v}")
        return v

    @classmethod
    def from_yaml(cls, yaml_str: str, dataset_dir: Optional[str] = None) -> "EvalTask":
        obj = yaml.safe_load(yaml_str)
        input_dataset_name = obj.get("input_dataset_name")
        if input_dataset_name:
            if dataset_dir:
                path = os.path.join(dataset_dir, input_dataset_name)
            else:
                path = input_dataset_name
            with open(path, "rb") as f:
                obj["input_dataset_binary"] = f.read()
        logger.debug(f"EvalTask.from_yaml obj: {obj}")
        task = cls(**obj)
        logger.debug(f"EvalTask.from_yaml task: {task}")
        return task

    def run_provider(self, data_list: list[Result], stop_event: threading.Event):
        """Run data list with batch"""
        batch_size = self.run_config.batch_size
        batch_data_list = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
        with ThreadPoolExecutor(max_workers=self.run_config.parallelism) as executor:
            # Submit tasks for execution
            futures = [
                executor.submit(self.batch_run, batch_data=batch_data) for batch_data in batch_data_list
            ]

            # Collect results from completed tasks
            for future in as_completed(futures):
                if stop_event.is_set():
                    return
                batch_result = future.result()
                yield from batch_result


    def run_eval(self, data_list: list[Result], stop_event: threading.Event, default_eval_llm: Optional[LLM] = None):
        """Eval data list with batch"""
        # TODO seperate eval run config
        batch_size = self.run_config.batch_size
        batch_data_list = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
        for evaluator in self.evaluators:
            with ThreadPoolExecutor(max_workers=self.run_config.parallelism) as executor:
                # Submit tasks for execution
                futures = [
                    executor.submit(
                        self.batch_eval,
                        evaluator=evaluator,
                        batch_data=batch_data,
                        default_eval_llm=default_eval_llm,
                    ) for batch_data in batch_data_list
                ]

                # Collect results from completed tasks
                for future in as_completed(futures):
                    if stop_event.is_set():
                        return
                    batch_result = future.result()
                    yield from batch_result

    def batch_run(self, batch_data: list[Result]) -> list[Result]:
        """Batch run data"""
        start = perf_counter()
        run_error = ""
        if self.provider is not None:
            inputs = [data.inputs for data in batch_data]
            try:
                # 1. 首先调用 LLM
                run_outputs = self.provider.batch_call(
                    inputs, timeout=self.run_config.timeout)
                logger.debug(f"provider call: {inputs} -> {run_outputs}")
                for i, data in enumerate(batch_data):
                    data.run.outputs = run_outputs[i]
            except Exception as e:
                logger.error(f"provider call failed: {e}", exc_info=True)
                run_error = str(e)

            for data in batch_data:
                data.run.error = run_error
                data.run.elapsed_secs = perf_counter() - start
        return batch_data

    def batch_eval(self,
                   evaluator: Evaluator,
                   batch_data: list[Result],
                   default_eval_llm: Optional[LLM] = None) -> list[Result]:
        start = perf_counter()
        run_error = ""
        for data in batch_data:
            data.evals[evaluator.name] = EvalResult()
        try:
            eval_outputs = evaluator.batch_call(
                batch_inputs=[data.inputs for data in batch_data],
                batch_outputs=[data.run.outputs for data in batch_data],
                timeout=self.run_config.timeout,
                default_llm=default_eval_llm,
            )
            for i, data in enumerate(batch_data):
                data.evals[evaluator.name].outputs = eval_outputs[i]
        except Exception as e:
            logger.warning(f"evaluator call failed: {e}", exc_info=True)
            run_error = str(e)

        for data in batch_data:
            data.evals[evaluator.name].error = run_error
            data.evals[evaluator.name].elapsed_secs = perf_counter() - start
        return batch_data

    def split_dataset(self) -> list[dict[str, Any]]:
        if self.input_dataset_name.endswith(".csv"):
            return self.split_csv_dataset()
        elif self.input_dataset_name.endswith(".jsonl"):
            return self.split_jsonl_dataset()
        else:
            raise ValueError(
                f"Invalid input_dataset_name: {self.input_dataset_name}")

    def split_csv_dataset(self) -> list[dict[str, Any]]:
        if not self.input_dataset_binary:
            return []
        data_list = []
        with io.StringIO(self.input_dataset_binary.decode("utf-8")) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data_list.append(row)
        return data_list

    def split_jsonl_dataset(self) -> list[dict[str, Any]]:
        if not self.input_dataset_binary:
            return []
        data_list = []
        for line in self.input_dataset_binary.decode("utf-8").split("\n"):
            if line.strip():
                data_list.append(json.loads(line.strip()))
        return data_list

    def input_dataset_pd(self, limit: int = 5) -> pd.DataFrame:
        data = self.split_dataset()
        if len(data) > limit:
            data = data[:limit]
        return pd.DataFrame(data)
