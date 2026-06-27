"""
MLflow context manager for pipeline run tracking.

Usage:
    with MLflowTracker(params={"model": ..., ...}) as tracker:
        # run pipeline
        tracker.log_prompt(template_path, prompt_version)
        tracker.log_eval_dataset(eval_set_path, golds)
        tracker.log_metrics({"weighted_f1": 0.85, ...})
        tracker.log_artifact("/path/to/results.jsonl")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import mlflow
import mlflow.openai
from mlflow import MlflowClient

# Patch the OpenAI SDK once at import time so every call — regardless of which
# run is active — is captured as a trace. Calling this inside start_run() would
# re-register the patch on every pipeline execution unnecessarily.
mlflow.openai.autolog(log_traces=True)

_EXPERIMENT_NAME = "pathology-extraction"
_PROMPT_REGISTRY_NAME = "pathology-extraction-prompt"
# Absolute path to the project root (two levels up from this file: src/pipeline/ → src/ → project root)
_PROJECT_ROOT = Path(__file__).parents[2].resolve()
_DEFAULT_TRACKING_URI = f"sqlite:///{_PROJECT_ROOT / 'mlflow.db'}"


def _get_or_register_prompt(
    client: MlflowClient, template: str, file_version: str
) -> Any:
    """Return existing PromptVersion for this file version or register a new one.

    MLflow uses auto-incremented integer versions internally. We tag each version
    with file_version (e.g. "v1.7") so the same template is never registered twice.
    """
    try:
        versions = client.search_prompt_versions(_PROMPT_REGISTRY_NAME)
        for v in versions:
            if v.tags and v.tags.get("file_version") == file_version:
                logger.debug("Reusing existing prompt version %s for %s", v.version, file_version)
                return v
    except Exception:
        pass  # prompt doesn't exist yet — fall through to register

    return client.register_prompt(
        name=_PROMPT_REGISTRY_NAME,
        template=template,
        commit_message=f"Prompt template {file_version}",
        tags={"file_version": file_version},
    )


class MLflowTracker:
    def __init__(self, params: dict[str, Any]) -> None:
        self._params = params
        self._run: mlflow.ActiveRun | None = None
        self._client: MlflowClient | None = None

    def __enter__(self) -> "MLflowTracker":
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(_EXPERIMENT_NAME)

        model = self._params.get("model", "unknown")
        prompt_version = self._params.get("prompt_version", "unknown")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        # colons in model string are invalid in run names
        run_name = f"{model.replace(':', '-')}__{prompt_version}__{ts}"

        self._run = mlflow.start_run(run_name=run_name)
        self._client = MlflowClient()
        mlflow.log_params(self._params)
        return self

    def log_prompt(self, template_path: str | Path, prompt_version: str) -> None:
        """Register prompt template in the MLflow Prompt Registry and link to run and experiment.

        If a version with this file_version tag already exists it is reused,
        so repeated runs with the same template never create duplicates.
        """
        if self._run is None or self._client is None:
            return
        try:
            template = Path(template_path).read_text(encoding="utf-8")
            pv = _get_or_register_prompt(self._client, template, prompt_version)
            # Link to the run (shows in run detail view)
            self._client.link_prompt_version_to_run(
                run_id=self._run.info.run_id, prompt=pv
            )
            # Link to the experiment so it appears in the experiment's Prompts tab
            self._client._link_prompt_to_experiment(
                prompt_version=pv,
                experiment_id=self._run.info.experiment_id,
            )
            logger.debug("Linked prompt %s (MLflow version %s) to run and experiment", prompt_version, pv.version)
        except Exception as exc:
            logger.warning("Could not register prompt in MLflow registry: %s", exc)

    def log_eval_dataset(self, path: str | Path, records: list[dict]) -> None:
        """Log the evaluation ground-truth dataset as an MLflow input.

        This populates the Datasets tab in the MLflow UI so you can trace which
        eval set was used for each run and compare across experiments.
        """
        if self._run is None or not records:
            return
        try:
            import pandas as pd

            df = pd.DataFrame(records)
            dataset = mlflow.data.from_pandas(
                df,
                source=str(path),
                name="eval_ground_truth",
            )
            mlflow.log_input(dataset, context="evaluation")
        except Exception as exc:
            logger.warning("Could not log eval dataset to MLflow: %s", exc)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._run is not None:
            mlflow.log_metrics(metrics)

    def log_artifact(self, path: str | Path) -> None:
        if self._run is None:
            return
        if Path(path).exists():
            mlflow.log_artifact(str(path))
        else:
            logger.warning("log_artifact: file not found, skipping: %s", path)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        mlflow.end_run()
        return False  # do not suppress exceptions

    @property
    def run_id(self) -> str | None:
        if self._run is not None:
            return self._run.info.run_id
        return None
