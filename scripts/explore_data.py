"""Generate reports/stat_report.md for the classification project."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "train.csv"
TEST = ROOT / "data" / "test.csv"
SAMPLE = ROOT / "data" / "sample_submission.csv"
REPORT = ROOT / "reports" / "stat_report.md"

LABELS = {
    0: "not a movie/TV review",
    1: "positive movie/TV review",
    2: "negative movie/TV review",
}
EXPECTED = {
    "train": ["ID", "TEXT", "LABEL"],
    "test": ["ID", "TEXT"],
    "sample_submission": ["ID", "LABEL"],
}
NUMERIC = ["char_count", "word_count", "exclamation_count", "question_count", "uppercase_ratio"]
BINARY = ["is_blank_text", "has_url", "has_html", "has_repeated_punctuation"]
TOKEN = r"(?u)\b\w[\w']+\b"


def log(msg: str) -> None:
    print(f"[explore_data] {msg}")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def pct(num: float, den: float) -> float:
    return 0.0 if den == 0 or pd.isna(den) else 100.0 * float(num) / float(den)


def as_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def normalize(series: pd.Series) -> pd.Series:
    return as_text(series).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4g}" if math.isfinite(float(value)) else str(value)
    return str(value)


def md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "_No rows._"

    shown = df.head(max_rows).copy() if max_rows else df.copy()
    cols = list(shown.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]

    for _, row in shown.iterrows():
        values = [cell(row[col]).replace("|", "\\|").replace("\n", " ") for col in cols]
        lines.append("| " + " | ".join(values) + " |")

    if max_rows and len(df) > max_rows:
        lines += ["", f"_Showing {max_rows} of {len(df)} rows._"]
    return "\n".join(lines)


# Load a required CSV while preserving ID values as strings.
def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")

    df = pd.read_csv(path, dtype={"ID": "string"}, low_memory=False)
    if "TEXT" in df.columns:
        df["TEXT"] = df["TEXT"].astype("string")
    return df


def label_value(value: object) -> object:
    if pd.isna(value):
        return value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return int(value) if value.is_integer() else value


def label_name(value: object) -> str:
    if pd.isna(value):
        return "missing/unparseable"
    return LABELS.get(label_value(value), "unexpected label")


# Count expected, unexpected, and missing training labels.
def label_distribution(train: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    rows = []

    for label, meaning in LABELS.items():
        count = int(labels.eq(label).sum())
        rows.append({"LABEL": label, "meaning": meaning, "count": count,
                     "proportion_pct": pct(count, len(train))})

    unexpected = labels[labels.notna() & ~labels.isin(list(LABELS))]
    for label, count in unexpected.value_counts().sort_index().items():
        rows.append({"LABEL": label_value(label), "meaning": "unexpected label",
                     "count": int(count), "proportion_pct": pct(count, len(train))})

    missing = int(labels.isna().sum())
    if missing:
        rows.append({"LABEL": "missing_or_unparseable",
                     "meaning": "missing or unparseable label",
                     "count": missing, "proportion_pct": pct(missing, len(train))})

    return pd.DataFrame(rows), unexpected


# Count missing values and blank strings in every input column.
def missing_report(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for split, df in datasets.items():
        for col in df.columns:
            series = df[col]
            is_text = series.dtype == object or pd.api.types.is_string_dtype(series)
            blanks = int(as_text(series).str.strip().eq("").sum()) if is_text else 0
            missing = int(series.isna().sum())
            rows.append({"split": split, "column": col,
                         "missing_count": missing, "missing_pct": pct(missing, len(df)),
                         "blank_count": blanks, "blank_pct": pct(blanks, len(df))})

    return pd.DataFrame(rows)


# Check ID uniqueness and sample-submission alignment.
def id_report(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for split, df in {"train": train, "test": test, "sample_submission": sample}.items():
        ids = df["ID"]
        for check, value, details in [
            ("row_count", len(df), "number of rows"),
            ("unique_id_count", int(ids.nunique(dropna=False)), "unique IDs including missing"),
            ("missing_id_count", int(ids.isna().sum()), "missing ID values"),
            ("duplicate_id_rows", int(ids.duplicated(keep=False).sum()),
             "rows with duplicated IDs"),
        ]:
            rows.append({"check": f"{split}_{check}", "value": value, "details": details})

    train_ids = as_text(train["ID"])
    test_ids = as_text(test["ID"])
    sample_ids = as_text(sample["ID"])
    rows += [
        {"check": "train_test_id_overlap_count",
         "value": len(set(train_ids) & set(test_ids)),
         "details": "shared IDs between train and test"},
        {"check": "sample_submission_row_count_matches_test",
         "value": len(sample) == len(test),
         "details": f"sample rows={len(sample)}, test rows={len(test)}"},
        {"check": "sample_submission_id_set_matches_test",
         "value": set(sample_ids) == set(test_ids),
         "details": "ID set equality, ignoring order"},
        {"check": "sample_submission_id_order_matches_test",
         "value": sample_ids.reset_index(drop=True).equals(test_ids.reset_index(drop=True)),
         "details": "exact ID order equality"},
    ]
    return pd.DataFrame(rows)


# Create compact surface-form text features for EDA tables.
def text_features(df: pd.DataFrame, split: str) -> pd.DataFrame:
    text = as_text(df["TEXT"])
    letters = text.str.count(r"[A-Za-z]").astype(float)
    capitals = text.str.count(r"[A-Z]").astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        uppercase_ratio = np.where(letters > 0, capitals / letters, 0.0)

    return pd.DataFrame({
        "ID": df["ID"],
        "split": split,
        "char_count": text.str.len().astype(float),
        "word_count": text.str.count(TOKEN).astype(float),
        "exclamation_count": text.str.count(r"!").astype(float),
        "question_count": text.str.count(r"\?").astype(float),
        "uppercase_ratio": uppercase_ratio,
        "is_blank_text": text.str.strip().eq(""),
        "has_url": text.str.contains(r"https?://|www\.", case=False, regex=True, na=False),
        "has_html": text.str.contains(r"<[^>]+>|&[a-z]+;", case=False, regex=True, na=False),
        "has_repeated_punctuation": text.str.contains(r"[!?.,;:]{3,}", regex=True, na=False),
    })


def num_stats(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0, "mean": np.nan, "min": np.nan,
                "median": np.nan, "p90": np.nan, "max": np.nan}
    return {"count": int(len(clean)), "mean": clean.mean(), "min": clean.min(),
            "median": clean.median(), "p90": clean.quantile(0.9), "max": clean.max()}


# Summarize numeric text features by split or label.
def summarize_numeric(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []

    for key, group in df.groupby(groups, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(groups, key))
        for col in NUMERIC:
            rows.append({**group_values, "feature": col, **num_stats(group[col])})

    return pd.DataFrame(rows)


# Summarize binary text features by training label.
def summarize_binary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for key, group in df.groupby(["LABEL", "label_meaning"], dropna=False):
        label, meaning = key
        for col in BINARY:
            count = int(group[col].fillna(False).astype(bool).sum())
            rows.append({"LABEL": label, "label_meaning": meaning, "feature": col,
                         "true_count": count, "true_pct": pct(count, len(group))})

    return pd.DataFrame(rows)


# Report where normalized duplicate text exists without listing examples.
def duplicate_summary(train: pd.DataFrame, test: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    train_text = normalize(train["TEXT"])
    test_text = normalize(test["TEXT"])
    train_text = train_text[train_text.ne("")]
    test_text = test_text[test_text.ne("")]

    train_dups = train_text.value_counts().loc[lambda counts: counts.gt(1)]
    test_dups = test_text.value_counts().loc[lambda counts: counts.gt(1)]
    shared = set(train_text.unique()) & set(test_text.unique())

    train_work = pd.DataFrame({"TEXT": train_text, "LABEL": labels.reindex(train_text.index)})
    conflicts = pd.DataFrame(columns=["row_count"])
    if not train_work.empty:
        grouped = train_work.groupby("TEXT").agg(
            row_count=("LABEL", "size"),
            unique_label_count=("LABEL", lambda x: int(x.dropna().nunique())),
        )
        conflicts = grouped[grouped["row_count"].gt(1) & grouped["unique_label_count"].gt(1)]

    return pd.DataFrame([
        {"location": "train", "check": "duplicate normalized text",
         "group_count": len(train_dups),
         "train_row_count": int(train_text.isin(train_dups.index).sum()),
         "test_row_count": "", "exists": bool(len(train_dups))},
        {"location": "train", "check": "duplicate normalized text with conflicting labels",
         "group_count": len(conflicts),
         "train_row_count": int(conflicts["row_count"].sum()) if not conflicts.empty else 0,
         "test_row_count": "", "exists": bool(len(conflicts))},
        {"location": "test", "check": "duplicate normalized text",
         "group_count": len(test_dups), "train_row_count": "",
         "test_row_count": int(test_text.isin(test_dups.index).sum()),
         "exists": bool(len(test_dups))},
        {"location": "train/test", "check": "shared normalized text",
         "group_count": len(shared), "train_row_count": int(train_text.isin(shared).sum()),
         "test_row_count": int(test_text.isin(shared).sum()), "exists": bool(len(shared))},
    ])


# Build all tables included in the Markdown report.
def build_tables(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    labels = pd.to_numeric(train["LABEL"], errors="coerce")
    datasets = {"train": train, "test": test, "sample_submission": sample}
    train_feats = text_features(train, "train")
    test_feats = text_features(test, "test")
    label_counts, unexpected = label_distribution(train, labels)

    train_labeled = train_feats.assign(
        LABEL=labels.reset_index(drop=True),
        label_meaning=labels.reset_index(drop=True).map(label_name),
    )
    label_integrity = pd.DataFrame([{
        "valid_label_rows": int(labels.isin(list(LABELS)).sum()),
        "missing_or_unparseable_rows": int(labels.isna().sum()),
        "unexpected_label_rows": int(unexpected.shape[0]),
        "unexpected_label_values": ", ".join(map(str, sorted(unexpected.dropna().unique()))),
        "all_training_labels_valid": bool(labels.notna().all() and unexpected.empty),
    }])

    columns = []
    for split, df in datasets.items():
        expected = EXPECTED[split]
        actual = list(df.columns)
        columns.append({"split": split, "expected_columns": ", ".join(expected),
                        "actual_columns": ", ".join(actual),
                        "matches_expected_order": actual == expected,
                        "missing_expected_columns": ", ".join(
                            c for c in expected if c not in actual
                        ),
                        "unexpected_columns": ", ".join(c for c in actual if c not in expected)})

    return {
        "overview": pd.DataFrame({"split": list(datasets),
                                  "rows": [len(df) for df in datasets.values()],
                                  "columns": [len(df.columns) for df in datasets.values()]}),
        "columns": pd.DataFrame(columns),
        "missing": missing_report(datasets),
        "ids": id_report(train, test, sample),
        "label_counts": label_counts,
        "label_integrity": label_integrity,
        "num_by_split": summarize_numeric(pd.concat([train_feats, test_feats]), ["split"]),
        "num_by_label": summarize_numeric(train_labeled, ["LABEL", "label_meaning"]),
        "binary_by_label": summarize_binary(train_labeled),
        "duplicate_summary": duplicate_summary(train, test, labels),
    }


def section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}"


# Render the report as tables with narrative text.
def report_text(tables: dict[str, pd.DataFrame]) -> str:
    label_map = pd.DataFrame({"LABEL": list(LABELS), "meaning": list(LABELS.values())})
    files = pd.DataFrame({"file": [rel(TRAIN), rel(TEST), rel(SAMPLE)]})

    integrity = "\n\n".join([
        "### Row counts and schema", md(tables["overview"]), md(tables["columns"]),
        "### Missing values and blank strings", md(tables["missing"]),
        "### ID and submission checks", md(tables["ids"], 50),
        "### Label integrity", md(tables["label_integrity"]),
    ])
    text_stats = "\n\n".join([
        "### Numeric features by split", md(tables["num_by_split"]),
        "### Numeric features by training label", md(tables["num_by_label"], 40),
        "### Binary features by training label", md(tables["binary_by_label"], 40),
    ])
    parts = [
        "# Statistical Data Report",
        "Generated by `scripts/explore_data.py`.",
        section("Task labels", md(label_map)),
        section("Input files", md(files)),
        section("Data integrity", integrity),
        section("Label distribution", md(tables["label_counts"])),
        section("Text statistics", text_stats),
        section("Duplicate summary", md(tables["duplicate_summary"])),
    ]
    return "\n\n".join(parts) + "\n"


# Run the report workflow from the project paths.
def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    log(f"Reading {rel(TRAIN)}, {rel(TEST)}, and {rel(SAMPLE)}")

    train = read_csv(TRAIN)
    test = read_csv(TEST)
    sample = read_csv(SAMPLE)

    log("Computing integrity checks and text statistics")
    tables = build_tables(train, test, sample)

    log(f"Writing {rel(REPORT)}")
    REPORT.write_text(report_text(tables), encoding="utf-8")
    log("Statistical report complete")


if __name__ == "__main__":
    main()
