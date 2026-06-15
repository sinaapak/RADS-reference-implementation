from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .pipeline import run_experiment


VARIANTS = (
    "full",
    "without_cross_attention",
    "without_text_transformer",
    "without_rads",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table 4 model ablations.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--price-column", required=True)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--headline-column", default="headline")
    parser.add_argument("--label-column", default="is_fake")
    args = parser.parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    rows = []
    for variant in VARIANTS:
        metrics = run_experiment(
            args.data,
            args.price_column,
            config,
            date_column=args.date_column,
            headline_column=args.headline_column,
            label_column=args.label_column,
            variant=variant,
        )
        rows.append({"variant": variant, **metrics})
    output = Path(config.output_dir) / args.price_column / "ablation_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
