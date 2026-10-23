"""Register one chunking configuration in the MLflow Model Registry.

    python scripts/register_best.py baseline-whole-article --note "why"

The registered "model" is a config bundle: a pyfunc model whose artifact is the
chunking configuration of one MLflow run (chunk_size, overlap, embedding model,
k, ...). It is logged on the SAME run that produced the metrics, its metrics are
copied onto the model version as tags, and the alias `production` is moved to
the new version. params.yaml stays what build_index.py reads; the registry is
the versioned record of which configuration was promoted and on what evidence.

The run must have faithfulness metrics (run run_experiments.py first). Running
the script twice creates a second version and moves the alias to it.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME = "arabic-legal-rag-chunking"
MODEL_NAME = "arabic-legal-rag-chunking-config"
ALIAS = "production"
TAGGED_METRICS = (
    "faithfulness",
    "faithfulness_n_scored",
    "faithfulness_n_questions",
    "refusal_rate",
    "hit_at_1_all",
    "hit_at_3_all",
    "n_chunks",
)


class ChunkingConfig(mlflow.pyfunc.PythonModel):
    """Returns the stored chunking configuration (the input is ignored)."""

    def load_context(self, context):
        path = Path(context.artifacts["config"])
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def predict(self, model_input, params=None):
        return [self.config]


def find_run(client, run_name):
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit(f"Experiment {EXPERIMENT_NAME!r} not found.")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
    )
    scored = [r for r in runs if "faithfulness_n_scored" in r.data.metrics]
    if not scored:
        raise SystemExit(f"No run named {run_name!r} with faithfulness metrics.")
    return scored[0]  # newest


def register(run, note):
    client = MlflowClient()
    config = dict(run.data.params)
    with tempfile.TemporaryDirectory() as tmp:
        config_path = Path(tmp) / "chunking_config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with mlflow.start_run(run_id=run.info.run_id):
            info = mlflow.pyfunc.log_model(
                name="chunking_config",
                python_model=ChunkingConfig(),
                artifacts={"config": str(config_path)},
                registered_model_name=MODEL_NAME,
                pip_requirements=["mlflow"],
            )
    version = str(info.registered_model_version)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    client.set_model_version_tag(MODEL_NAME, version, "source_run_id", run.info.run_id)
    client.set_model_version_tag(
        MODEL_NAME, version, "source_run_name", run.info.run_name
    )
    for key in TAGGED_METRICS:
        if key in run.data.metrics:
            value = f"{run.data.metrics[key]:.4g}"
            client.set_model_version_tag(MODEL_NAME, version, key, value)
    if note:
        client.update_model_version(MODEL_NAME, version, description=note)
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_name", help="MLflow run name, e.g. baseline-whole-article")
    parser.add_argument("--note", default="", help="why this config is promoted")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    run = find_run(MlflowClient(), args.run_name)
    version = register(run, args.note)

    loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    print(f"registered {MODEL_NAME} version {version}, alias {ALIAS!r}")
    print("loaded from alias:", loaded.predict([0])[0])


if __name__ == "__main__":
    main()
