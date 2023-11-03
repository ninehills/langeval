import csv
import io
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field, validator

from langeval.evaluators import Evaluator
from langeval.models import LLM
from langeval.providers import Provider

logger = logging.getLogger(__name__)


class Result(BaseModel):
    """Result for one data"""

    inputs: dict[str, Any]
    run_error: str = ""
    run_outputs: dict[str, Any] = {}
    run_elapsed_secs: float = 0
    eval_error: str = ""
    eval_outputs: dict[str, Any] = {}
    eval_elapsed_secs: float = 0

    def to_jsonl(self) -> str:
        return json.dumps(self.dict(), ensure_ascii=False) + "\n"

    @classmethod
    def from_json(cls, json_str: str) -> "Result":
        return cls(**json.loads(json_str))


class TaskRunConfig(BaseModel):
    parallelism: int = Field(default=1, ge=1, le=10)
    timeout: int = Field(default=30, ge=1, le=600)
    rounds: int = Field(default=1, ge=1, le=5)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.dict(exclude_unset=True), encoding="utf-8", allow_unicode=True).decode("utf-8")

    @staticmethod
    def from_yaml(yaml_str: str) -> "TaskRunConfig":
        try:
            return TaskRunConfig(**yaml.safe_load(yaml_str))
        except Exception as e:
            raise ValueError(f"Invalid yaml: {e}") from e


class EvalTask(BaseModel):
    ## config
    provider: Optional[Provider] = None
    input_dataset_binary: Optional[bytes] = None
    # Only jsonl, csv supported
    input_dataset_name: str = Field(..., min_length=1, max_length=255)
    # evaluator
    evaluator: Optional[Evaluator] = None
    # Run config
    run_config: TaskRunConfig

    class Config:
        validate_assignment = True

    @validator("input_dataset_name")
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
        return cls(**obj)

    def run(self, data_list: list[dict[str, Any]], stop_event: threading.Event, default_eval_llm: Optional[LLM] = None):
        """Run data list and return results"""
        with ThreadPoolExecutor(max_workers=self.run_config.parallelism) as executor:
            # Submit tasks for execution
            futures = [
                executor.submit(self.run_one_data, data=data, default_eval_llm=default_eval_llm) for data in data_list
            ]

            # Collect results from completed tasks
            results = (future.result() for future in as_completed(futures))

            for result in results:
                if stop_event.is_set():
                    return
                yield result

    def run_one_data(self, data: dict[str, Any], default_eval_llm: Optional[LLM] = None) -> Result:
        """Run one data and return result"""
        start = datetime.utcnow()
        result = Result(inputs=data)
        if self.provider is not None:
            try:
                # 1. 首先调用 LLM
                result.run_outputs = self.provider.call(data, timeout=self.run_config.timeout)
                logger.debug(f"llm completion: {data} -> {result.run_outputs}")
            except Exception as e:
                logger.debug(f"llm call failed: {e}", exc_info=True)
                result.run_error = str(e)
            result.run_elapsed_secs = (datetime.utcnow() - start).total_seconds()
            if result.run_error != "":
                return result
        # 否则为仅评估不运行
        # 2. 然后调用 evaluator
        start = datetime.utcnow()
        if self.evaluator is not None:
            try:
                result.eval_outputs = self.evaluator.call(
                    inputs=data,
                    outputs=result.run_outputs,
                    timeout=self.run_config.timeout,
                    default_llm=default_eval_llm,
                )
            except Exception as e:
                logger.debug(f"evaluator call failed: {e}", exc_info=True)
                result.eval_error = str(e)
            result.eval_elapsed_secs = (datetime.utcnow() - start).total_seconds()
        return result

    def split_dataset(self) -> list[dict[str, Any]]:
        if self.input_dataset_name.endswith(".csv"):
            return self.split_csv_dataset()
        elif self.input_dataset_name.endswith(".jsonl"):
            return self.split_jsonl_dataset()
        else:
            raise ValueError(f"Invalid input_dataset_name: {self.input_dataset_name}")

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
