from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np

ROOT = Path("./score_pass_1")


def iter_records(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def summarize_file(path: Path) -> dict | None:
    scores, toks = [], []
    for rec in iter_records(path):
        scores.extend(rec.get("score", []))
        toks.extend(rec.get("token_len", []))
    if not scores or not toks:
        return None
    # scores may be bools; convert to int for averaging
    scores_int = [int(s) for s in scores]
    return {
        "file": str(path.relative_to(ROOT)),
        "n": len(scores_int),
        "avg_score": sum(scores_int) / len(scores_int) * 100,
        "avg_token_len": sum(toks) / len(toks),
    }


def aggregate_by_model(summaries: list[dict]) -> tuple[list[str], dict[str, dict[str, tuple[float, float]]]]:
    datasets: set[str] = set()
    table: dict[str, dict[str, tuple[float, float]]] = {}
    for s in summaries:
        parts = Path(s["file"]).parts  # e.g., arm-team/ARM-7B/gsm8k/...
        if len(parts) < 3:
            continue
        model = parts[-3]
        dataset = parts[-2]
        datasets.add(dataset)
        table.setdefault(model, {})[dataset] = (s["avg_score"], s["avg_token_len"])
    return sorted(datasets), table


def main() -> None:
    dataset_list = [
        # "gsm8k", "olympiadbench",
         "math500", "amc23", "aime24", "aime25"
    ]

    exclude_model_list = [
        "ARM-14B",
        "ARM-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
        "alpha_0.05_DeepSeek-R1-Distill-Qwen-1.5B",
    ]
    
    summaries = []
    for jsonl in ROOT.rglob("*.jsonl"):
        # Only process files from specified datasets
        parts = jsonl.relative_to(ROOT).parts
        model_name = parts[-3] if len(parts) >= 3 else None
        if model_name in exclude_model_list:
            continue
        if len(parts) >= 3 and parts[-2] in dataset_list:
            summary = summarize_file(jsonl)
            if summary:
                summaries.append(summary)
    datasets, table = aggregate_by_model(summaries)

    rows = []
    for model in sorted(table):
        row = {"model": model}
        for ds in datasets:
            score, tok = table[model].get(ds, (None, None))
            row[f"{ds}-acc"] = np.round(score, 1) if score is not None else None
            row[f"{ds}-len"] = np.round(tok, 0) if tok is not None else None
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
