from __future__ import annotations

import argparse
import json

from .config import ExperimentConfig
from .pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the manuscript-aligned RADS stock forecasting experiment."
    )
    parser.add_argument("--data", required=True, help="CSV or Excel input file")
    parser.add_argument("--price-column", required=True)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--headline-column", default="headline")
    parser.add_argument("--label-column", default="is_fake")
    parser.add_argument(
        "--variant",
        default="full",
        choices=[
            "full",
            "without_cross_attention",
            "without_text_transformer",
            "without_rads",
        ],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    metrics = run_experiment(
        args.data,
        args.price_column,
        config,
        date_column=args.date_column,
        headline_column=args.headline_column,
        label_column=args.label_column,
        variant=args.variant,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
