"""Export the latest faithfulness-scored run of each config to a Markdown table.

    python scripts/export_report.py     # writes reports/chunking_experiments.md
"""

import os
from pathlib import Path

import mlflow

EXPERIMENT_NAME = "arabic-legal-rag-chunking"
OUT = Path("reports/chunking_experiments.md")
HEADER = (
    "| Run | chunk_size | overlap | chunks | hit@1 all | hit@3 all | hit@3 ar "
    "| hit@3 en | faithfulness | scored / asked | refusal rate | judge |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def fmt(value, spec=".2f"):
    return "n/a" if value != value else format(value, spec)  # NaN != NaN


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME], order_by=["attributes.start_time ASC"]
    )
    if "metrics.faithfulness_n_scored" not in runs.columns:
        raise SystemExit("No run has faithfulness metrics yet; run run_experiments.py.")
    runs = runs.dropna(subset=["metrics.faithfulness_n_scored"])
    # Old reruns of the same config stay in MLflow; the table shows the newest one.
    latest = runs.drop_duplicates(subset="tags.mlflow.runName", keep="last")

    rows = []
    for _, r in latest.iterrows():
        rows.append(
            f"| {r['tags.mlflow.runName']} | {r['params.chunk_size']} "
            f"| {r['params.overlap']} | {int(r['metrics.n_chunks'])} "
            f"| {fmt(r['metrics.hit_at_1_all'])} | {fmt(r['metrics.hit_at_3_all'])} "
            f"| {fmt(r['metrics.hit_at_3_ar'])} | {fmt(r['metrics.hit_at_3_en'])} "
            f"| {fmt(r['metrics.faithfulness'])} "
            f"| {int(r['metrics.faithfulness_n_scored'])} / "
            f"{int(r['metrics.faithfulness_n_questions'])} "
            f"| {fmt(r['metrics.refusal_rate'])} | {r['params.judge_model']} |"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
