#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import pandas as pd


def detect_delimiter(csv_path: Path, fallback: str = ",") -> str:
    sample = csv_path.read_text(encoding="utf-8", errors="replace")[:10000]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except csv.Error:
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only columns needed for BERTScore evaluation."
    )
    parser.add_argument(
        "--input",
        default="case_benchmark_results.csv",
        help="Input CSV path (default: case_benchmark_results.csv)",
    )
    parser.add_argument(
        "--output",
        default="case_benchmark_results_bertscore_input.csv",
        help="Output CSV path (default: case_benchmark_results_bertscore_input.csv)",
    )
    parser.add_argument(
        "--keep-cols",
        nargs="+",
        default=["case_id", "expected_answer", "predicted_answer"],
        help="Columns to keep (default: case_id expected_answer predicted_answer)",
    )
    parser.add_argument(
        "--output-sep",
        default=";",
        help="Output separator (default: ;)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    input_sep = detect_delimiter(input_path)
    print(f"Detected input delimiter: {repr(input_sep)}")

    df = pd.read_csv(
        input_path,
        sep=input_sep,
        dtype=str,
        keep_default_na=False,
        engine="python",
        on_bad_lines="skip",
    )
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    missing_cols = [col for col in args.keep_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}"
        )

    cleaned_df = df[args.keep_cols].copy()
    cleaned_df.to_csv(output_path, sep=args.output_sep, index=False)

    print(f"Rows kept: {len(cleaned_df)}")
    print(f"Columns kept: {args.keep_cols}")
    print(f"Saved cleaned CSV to: {output_path}")


if __name__ == "__main__":
    main()
