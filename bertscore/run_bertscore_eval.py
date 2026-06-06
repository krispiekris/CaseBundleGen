#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from bert_score import score


DEFAULT_INPUT_CSV = Path("case_benchmark_results_bertscore_input.csv")
DEFAULT_SEP = ";"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace('/', '_').replace('-', '_').replace('.', '_')


def rounded(value: float, decimals: int) -> float:
    return round(float(value), decimals)


def patch_bertscore_tokenizer_max_length(max_length: int = 512) -> None:
    """
    Guard against tokenizer overflow for some HF models that expose an
    extremely large model_max_length sentinel value.
    """
    original_get_tokenizer = score.__globals__["get_tokenizer"]

    def safe_get_tokenizer(model_type: str, use_fast: bool = False):
        tokenizer = original_get_tokenizer(model_type, use_fast)
        current_max_len = getattr(tokenizer, "model_max_length", None)

        if isinstance(current_max_len, int) and current_max_len > 100000:
            tokenizer.model_max_length = max_length
            if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
                tokenizer.init_kwargs["model_max_length"] = max_length

        return tokenizer

    score.__globals__["get_tokenizer"] = safe_get_tokenizer


def load_csv_robust(input_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(input_path, sep=DEFAULT_SEP, dtype=str, keep_default_na=False)
    except pd.errors.ParserError as exc:
        print(
            "Strict CSV parsing failed; retrying with tolerant parser "
            f"(skipping malformed rows). Error: {exc}"
        )
        return pd.read_csv(
            input_path,
            sep=DEFAULT_SEP,
            dtype=str,
            keep_default_na=False,
            engine="python",
            on_bad_lines="skip",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BERTScore on a semicolon-separated CSV and append per-model "
            "precision/recall/f1 columns."
        )
    )
    parser.add_argument(
        "--output",
        default="case_benchmark_results_bertscore_multi.csv",
        help="Output CSV path (default: case_benchmark_results_bertscore_multi.csv)",
    )
    parser.add_argument(
        "--case-summary-output",
        default=None,
        help=(
            "Optional case-level summary CSV path. "
            "If omitted, uses <output_stem>_by_case.csv"
        ),
    )
    parser.add_argument(
        "--final-summary-output",
        default=None,
        help=(
            "Optional overall summary CSV path. "
            "If omitted, uses <output_stem>_final.csv"
        ),
    )
    parser.add_argument(
        "--id-col",
        default="case_id",
        help="ID column name (default: case_id)",
    )
    parser.add_argument(
        "--expected-col",
        default="expected_answer",
        help="Reference/expected answer column name (default: expected_answer)",
    )
    parser.add_argument(
        "--predicted-col",
        default="predicted_answer",
        help="Candidate/predicted answer column name (default: predicted_answer)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language used for baseline rescaling (default: en)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for BERTScore inference (default: 16)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["roberta-large", "microsoft/deberta-xlarge-mnli"],
        help=(
            "One or more model names to evaluate. "
            "Default: roberta-large microsoft/deberta-xlarge-mnli"
        ),
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Number of decimal places for numeric values in output CSVs (default: 4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_bertscore_tokenizer_max_length()

    input_path = DEFAULT_INPUT_CSV
    output_path = Path(args.output)
    case_summary_output_path = (
        Path(args.case_summary_output)
        if args.case_summary_output
        else output_path.with_name(f"{output_path.stem}_by_case{output_path.suffix}")
    )
    final_summary_output_path = (
        Path(args.final_summary_output)
        if args.final_summary_output
        else output_path.with_name(f"{output_path.stem}_final{output_path.suffix}")
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}. "
            "Create it first with trim_bertscore_columns.py."
        )

    print(f"Using input file: {input_path}")
    df = load_csv_robust(input_path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    for required_col in [args.id_col, args.expected_col, args.predicted_col]:
        if required_col not in df.columns:
            raise ValueError(
                f"Missing required column '{required_col}'. "
                f"Found columns: {list(df.columns)}"
            )

    refs = df[args.expected_col].fillna("").astype(str).tolist()
    cands = df[args.predicted_col].fillna("").astype(str).tolist()

    print(f"Rows to score: {len(df)}")

    for model_name in args.models:
        print(f"Scoring with model: {model_name}")
        p_scores, r_scores, f1_scores = score(
            cands,
            refs,
            model_type=model_name,
            lang=args.lang,
            rescale_with_baseline=True,
            batch_size=args.batch_size,
            verbose=True,
        )

        suffix = sanitize_model_name(model_name)
        df[f"bertscore_precision__{suffix}"] = p_scores.tolist()
        df[f"bertscore_recall__{suffix}"] = r_scores.tolist()
        df[f"bertscore_f1__{suffix}"] = f1_scores.tolist()

        print(
            f"{model_name} mean P/R/F1: "
            f"{df[f'bertscore_precision__{suffix}'].mean():.4f} / "
            f"{df[f'bertscore_recall__{suffix}'].mean():.4f} / "
            f"{df[f'bertscore_f1__{suffix}'].mean():.4f}"
        )

    # Round float columns for cleaner export into spreadsheets.
    float_cols = df.select_dtypes(include=["float32", "float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].round(args.decimals)

    df.to_csv(output_path, sep=DEFAULT_SEP, index=False)
    print(f"Saved output to: {output_path}")

    score_cols = [col for col in df.columns if col.startswith("bertscore_")]
    case_agg_ops = ["mean", "min", "max", "std"]
    case_summary_df = df.groupby(args.id_col, as_index=False)[score_cols].agg(case_agg_ops)

    # Flatten the MultiIndex columns from groupby aggregations.
    flattened_cols = []
    for col in case_summary_df.columns:
        if isinstance(col, tuple):
            base, agg = col
            if base == args.id_col:
                flattened_cols.append(args.id_col)
            else:
                flattened_cols.append(f"{base}__{agg}")
        else:
            flattened_cols.append(col)
    case_summary_df.columns = flattened_cols
    case_summary_df = case_summary_df.sort_values(args.id_col)
    case_summary_df = case_summary_df.fillna(0.0)

    case_counts_df = (
        df.groupby(args.id_col, as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
        .sort_values(args.id_col)
    )
    case_summary_df = case_counts_df.merge(case_summary_df, on=args.id_col, how="left")
    case_summary_for_stats = case_summary_df.copy()
    case_float_cols = case_summary_df.select_dtypes(include=["float32", "float64"]).columns
    if len(case_float_cols) > 0:
        case_summary_df[case_float_cols] = case_summary_df[case_float_cols].round(args.decimals)
    case_summary_df.to_csv(case_summary_output_path, sep=DEFAULT_SEP, index=False)
    print(f"Saved case summary to: {case_summary_output_path}")

    final_columns = [
        "",
        "total_rows",
        "total_cases",
        "precision_mean",
        "precision_min",
        "precision_max",
        "precision_std",
        "recall__mean",
        "recall__min",
        "recall__max",
        "recall__std",
        "f1__mean",
        "f1__min",
        "f1__max",
        "f1__std",
    ]

    final_rows = [
        {
            "": "",
            "total_rows": str(int(len(df))),
            "total_cases": str(int(df[args.id_col].nunique())),
        }
    ]

    for model_name in args.models:
        suffix = sanitize_model_name(model_name)
        p_col = f"bertscore_precision__{suffix}"
        r_col = f"bertscore_recall__{suffix}"
        f_col = f"bertscore_f1__{suffix}"

        final_rows.append(
            {
                "": f"{suffix}_per_row",
                "precision_mean": rounded(df[p_col].mean(), args.decimals),
                "precision_min": rounded(df[p_col].min(), args.decimals),
                "precision_max": rounded(df[p_col].max(), args.decimals),
                "precision_std": rounded(df[p_col].std(ddof=0), args.decimals),
                "recall__mean": rounded(df[r_col].mean(), args.decimals),
                "recall__min": rounded(df[r_col].min(), args.decimals),
                "recall__max": rounded(df[r_col].max(), args.decimals),
                "recall__std": rounded(df[r_col].std(ddof=0), args.decimals),
                "f1__mean": rounded(df[f_col].mean(), args.decimals),
                "f1__min": rounded(df[f_col].min(), args.decimals),
                "f1__max": rounded(df[f_col].max(), args.decimals),
                "f1__std": rounded(df[f_col].std(ddof=0), args.decimals),
            }
        )

        p_case_col = f"bertscore_precision__{suffix}__mean"
        r_case_col = f"bertscore_recall__{suffix}__mean"
        f_case_col = f"bertscore_f1__{suffix}__mean"
        final_rows.append(
            {
                "": f"{suffix}_per_case",
                "precision_mean": rounded(case_summary_for_stats[p_case_col].mean(), args.decimals),
                "precision_min": rounded(case_summary_for_stats[p_case_col].min(), args.decimals),
                "precision_max": rounded(case_summary_for_stats[p_case_col].max(), args.decimals),
                "precision_std": rounded(case_summary_for_stats[p_case_col].std(ddof=0), args.decimals),
                "recall__mean": rounded(case_summary_for_stats[r_case_col].mean(), args.decimals),
                "recall__min": rounded(case_summary_for_stats[r_case_col].min(), args.decimals),
                "recall__max": rounded(case_summary_for_stats[r_case_col].max(), args.decimals),
                "recall__std": rounded(case_summary_for_stats[r_case_col].std(ddof=0), args.decimals),
                "f1__mean": rounded(case_summary_for_stats[f_case_col].mean(), args.decimals),
                "f1__min": rounded(case_summary_for_stats[f_case_col].min(), args.decimals),
                "f1__max": rounded(case_summary_for_stats[f_case_col].max(), args.decimals),
                "f1__std": rounded(case_summary_for_stats[f_case_col].std(ddof=0), args.decimals),
            }
        )

    final_summary_df = pd.DataFrame(final_rows, columns=final_columns).fillna("")
    final_summary_df.to_csv(final_summary_output_path, sep=DEFAULT_SEP, index=False)
    print(f"Saved final summary to: {final_summary_output_path}")


if __name__ == "__main__":
    main()
