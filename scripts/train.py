from __future__ import annotations

import html
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

# The project root is fixed by the scripts/train.py location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PICKLE_MODULE_NAME = "scripts.train"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts

setattr(scripts, "train", sys.modules[__name__])
sys.modules[PICKLE_MODULE_NAME] = sys.modules[__name__]

RANDOM_SEED = 539
MODEL_NAME = "tfidf_hierarchical_sgd_hinge_refined_v2"
VALIDATION_SIZE = 0.20
LABELS = [0, 1, 2]
LABEL_MEANINGS = {
    0: "not a movie/TV review",
    1: "positive movie/TV review",
    2: "negative movie/TV review",
}

WORD_MIN_DF = 2
WORD_MAX_DF = 0.95
CHAR_ANALYZER = "char_wb"
NEGATION_SCOPE_WINDOW = 4

# Ideal hyperparameters were searched for earlier.
STAGE1 = {
    "task": "label 0 vs review",
    "word": (1, 2),
    "char": (3, 5),
    "char_min_df": 3,
    "loss": "hinge",
    "alpha": 2e-5,
    "surface": 0.5,
    "negation": False,
}
STAGE2 = {
    "task": "label 1 vs label 2",
    "word": (1, 3),
    "char": (3, 6),
    "char_min_df": 2,
    "loss": "hinge",
    "alpha": 5e-5,
    "surface": 1.0,
    "negation": True,
}
SGD_SETTINGS = {
    "penalty": "l2",
    "class_weight": None,
    "max_iter": 10000,
    "tol": 1e-5,
    "n_iter_no_change": 10,
    "average": True,
    "n_jobs": -1,
}
SURFACE_FEATURE_NAMES = [
    "log_char_count", "log_word_count", "sentence_punct_count", "exclamation_count",
    "question_count", "uppercase_ratio", "digit_ratio", "punct_ratio",
    "has_repeated_punctuation",
]

WHITESPACE_RE = re.compile(r"\s+")
BR_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", flags=re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
INVISIBLE_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
MIXED_PUNCT_RE = re.compile(r"(?:!\?|\?!)[!?]*")
REPEAT_EXCLAMATION_RE = re.compile(r"!{2,}")
REPEAT_QUESTION_RE = re.compile(r"\?{2,}")
REPEATED_PUNCT_SURFACE_RE = re.compile(r"([!?.,;:])\1{1,}|(?:!\?|\?!)[!?]*")
ELONGATED_RE = re.compile(r"([A-Za-z])\1{2,}")
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
NEGATION_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
WORD_TOKEN_FULL_RE = re.compile(r"\w+", flags=re.UNICODE)
RATING_OUT_OF_RE = re.compile(r"\b(10|[0-9](?:\.\d+)?)\s*out\s+of\s+10\b", flags=re.I)
RATING_SLASH_RE = re.compile(r"\b(10|[0-9](?:\.\d+)?)\s*/\s*10\b(?!\s*/)", flags=re.I)
STAR_DIGIT_RE = re.compile(r"\b([0-5](?:\.\d+)?)\s*-?\s*(?:star|stars)\b", flags=re.I)
STAR_WORD_RE = re.compile(r"\b(zero|one|two|three|four|five)\s*-?\s*(?:star|stars)\b", flags=re.I)
GRADE_RE = re.compile(r"(?<!\w)(a\+|a-|b\+|f-|d-)(?!\w)", flags=re.I)
TEXT_TRANSLATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u00a0": " ",
    "\u202f": " ", "\u2007": " ", "\u2010": " - ", "\u2011": " - ",
    "\u2012": " - ", "\u2013": " - ", "\u2014": " - ", "\u2015": " - ",
    "\u2212": " - ", "\u2026": " ... ",
})
CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "can not", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "here's": "here is", "how's": "how is", "i'd": "i would",
    "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'd": "it would", "it'll": "it will", "it's": "it is", "let's": "let us",
    "mightn't": "might not", "mustn't": "must not", "needn't": "need not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would", "we'll": "we will",
    "we're": "we are", "we've": "we have", "weren't": "were not", "what's": "what is",
    "when's": "when is", "where's": "where is", "who's": "who is", "why's": "why is",
    "won't": "will not", "wouldn't": "would not", "y'all": "you all", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have",
}
CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)
GENERIC_NT_RE = re.compile(r"\b([A-Za-z]+)n't\b", flags=re.IGNORECASE)
NEGATION_TRIGGERS = {"not", "no", "never", "cannot", "without", "hardly", "barely", "rarely", "neither", "nor"}
NEGATION_BOUNDARIES = {".", "!", "?", ";", ":"}


class CleanTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_negation_scope: bool = False) -> None:
        self.use_negation_scope = use_negation_scope

    def fit(self, X: Iterable[object], y: Iterable[object] | None = None) -> "CleanTextTransformer":
        return self

    def transform(self, X: Iterable[object]) -> list[str]:
        return [clean_model_text(value, self.use_negation_scope) for value in values_from_text_input(X)]


class SurfaceFeaturesTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[object], y: Iterable[object] | None = None) -> "SurfaceFeaturesTransformer":
        return self

    def transform(self, X: Iterable[object]) -> np.ndarray:
        return np.asarray([surface_features(value) for value in values_from_text_input(X)], dtype=np.float32)

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(SURFACE_FEATURE_NAMES, dtype=object)


class HierarchicalSGDClassifier(BaseEstimator, ClassifierMixin):
    """Two-stage classifier: non-review detection, then positive/negative sentiment."""

    def fit(self, X: Iterable[object], y: Iterable[object]) -> "HierarchicalSGDClassifier":
        X_series = to_text_series(X)
        y_series = pd.Series(y).astype(int).reset_index(drop=True)
        if len(X_series) != len(y_series):
            raise ValueError("X and y lengths differ.")
        if set(y_series.unique()) != set(LABELS):
            raise ValueError("Training labels must contain 0, 1, and 2.")
        review_mask = y_series.isin([1, 2])
        self.stage1_pipeline_ = build_stage_pipeline(STAGE1)
        self.stage1_pipeline_.fit(X_series, (y_series != 0).astype(int))
        self.stage2_pipeline_ = build_stage_pipeline(STAGE2)
        self.stage2_pipeline_.fit(X_series.loc[review_mask], y_series.loc[review_mask])
        self.classes_ = np.array(LABELS, dtype=int)
        return self

    def predict(self, X: Iterable[object]) -> np.ndarray:
        X_series = to_text_series(X)
        stage1_pred = np.asarray(self.stage1_pipeline_.predict(X_series)).astype(int)
        predictions = np.zeros(len(stage1_pred), dtype=int)
        review_indices = np.flatnonzero(stage1_pred == 1)
        if len(review_indices):
            predictions[review_indices] = self.stage2_pipeline_.predict(
                X_series.iloc[review_indices]
            ).astype(int)
        return predictions


CleanTextTransformer.__module__ = PICKLE_MODULE_NAME
SurfaceFeaturesTransformer.__module__ = PICKLE_MODULE_NAME
HierarchicalSGDClassifier.__module__ = PICKLE_MODULE_NAME


def scalar_to_text(value: object) -> str:
    if value is None:
        return ""
    try:
        return "" if pd.isna(value) else str(value)
    except (TypeError, ValueError):
        return str(value)


def values_from_text_input(X: Iterable[object]) -> list[object]:
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("Expected one text column.")
        return X.iloc[:, 0].tolist()
    if isinstance(X, pd.Series):
        return X.tolist()
    return list(X)


def to_text_series(X: Iterable[object]) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("Expected one text column.")
        return X.iloc[:, 0].reset_index(drop=True)
    if isinstance(X, pd.Series):
        return X.reset_index(drop=True)
    return pd.Series(list(X))


def make_duplicate_key(value: object) -> str:
    return WHITESPACE_RE.sub(" ", scalar_to_text(value).strip()).casefold()


def expand_contractions(text: str) -> str:
    text = CONTRACTION_RE.sub(lambda match: CONTRACTIONS[match.group(0).casefold()], text)
    return GENERIC_NT_RE.sub(lambda match: f"{match.group(1)} not", text)


def rating_token(score: float, total: float) -> str:
    ratio = score / total if total else 0.0
    if ratio >= 0.70:
        return " __rating_high__ "
    if ratio <= 0.40:
        return " __rating_low__ "
    return " __rating_mid__ "


def normalize_ratings(text: str) -> str:
    words = {"zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0}
    text = RATING_OUT_OF_RE.sub(lambda match: rating_token(float(match.group(1)), 10.0), text)
    text = RATING_SLASH_RE.sub(lambda match: rating_token(float(match.group(1)), 10.0), text)
    text = STAR_DIGIT_RE.sub(lambda match: rating_token(float(match.group(1)), 5.0), text)
    text = STAR_WORD_RE.sub(lambda m: rating_token(words[m.group(1).casefold()], 5.0), text)
    return GRADE_RE.sub(
        lambda m: " __grade_high__ "
        if m.group(1).casefold() in {"a+", "a-", "b+"}
        else " __grade_low__ ",
        text,
    )


def base_text_without_markup(value: object) -> str:
    text = scalar_to_text(value)
    text = html.unescape(text)
    text = BR_TAG_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(TEXT_TRANSLATION)
    text = INVISIBLE_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def mark_negation_scope(text: str) -> str:
    output: list[str] = []
    remaining = 0
    for token in NEGATION_TOKEN_RE.findall(text):
        lower = token.casefold()
        is_word = bool(WORD_TOKEN_FULL_RE.fullmatch(token))
        if token in NEGATION_BOUNDARIES:
            remaining = 0
            output.append(token)
        elif is_word:
            trigger = lower in NEGATION_TRIGGERS
            if remaining > 0 and not trigger:
                output.append(f"{token}_NEG")
                remaining -= 1
            else:
                output.append(token)
            if trigger:
                remaining = NEGATION_SCOPE_WINDOW
        else:
            output.append(token)
    return WHITESPACE_RE.sub(" ", " ".join(output)).strip()


def clean_model_text(value: object, use_negation_scope: bool = False) -> str:
    text = base_text_without_markup(value)
    text = EMAIL_RE.sub(" __email__ ", text)
    text = URL_RE.sub(" __url__ ", text)
    text = expand_contractions(text)
    text = normalize_ratings(text)
    text = MIXED_PUNCT_RE.sub(" __mixed_punct__ ", text)
    text = REPEAT_EXCLAMATION_RE.sub(" __repeat_exclamation__ ", text)
    text = REPEAT_QUESTION_RE.sub(" __repeat_question__ ", text)
    text = ELONGATED_RE.sub(r"\1\1", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return mark_negation_scope(text) if use_negation_scope else text


def surface_features(value: object) -> list[float]:
    text = base_text_without_markup(value)
    char_count = len(text)
    word_count = len(TOKEN_RE.findall(text))
    alpha_count = sum(char.isalpha() for char in text)
    upper_count = sum(char.isalpha() and char.isupper() for char in text)
    digit_count = sum(char.isdigit() for char in text)
    punct_count = sum(unicodedata.category(char).startswith("P") for char in text)
    char_den = float(char_count) if char_count else 1.0
    alpha_den = float(alpha_count) if alpha_count else 1.0
    return [
        float(np.log1p(char_count)),
        float(np.log1p(word_count)),
        float(sum(char in ".!?" for char in text)),
        float(text.count("!")),
        float(text.count("?")),
        float(upper_count / alpha_den),
        float(digit_count / char_den),
        float(punct_count / char_den),
        float(bool(REPEATED_PUNCT_SURFACE_RE.search(text))),
    ]


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def require_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    actual = list(df.columns)
    if actual != expected:
        raise ValueError(f"{name} columns must be {expected}; found {actual}.")


def validate_id_column(df: pd.DataFrame, name: str) -> None:
    missing = int(df["ID"].isna().sum())
    blank = int(df["ID"].map(lambda value: scalar_to_text(value).strip() == "").sum())
    duplicate = int(df["ID"].duplicated(keep=False).sum())
    if missing or blank or duplicate:
        raise ValueError(f"{name} invalid IDs: missing={missing}, blank={blank}, duplicates={duplicate}.")


def validate_labels(train: pd.DataFrame) -> pd.DataFrame:
    labels = pd.to_numeric(train["LABEL"], errors="coerce")
    if labels.isna().any():
        raise ValueError(f"train.csv has {int(labels.isna().sum())} invalid LABEL values.")
    rounded = np.rint(labels.to_numpy(dtype=float))
    if not np.allclose(labels.to_numpy(dtype=float), rounded):
        raise ValueError("train.csv LABEL values must be integers.")
    found = set(rounded.astype(int).tolist())
    if not found.issubset(LABELS):
        raise ValueError(f"Unexpected LABEL values: {sorted(found - set(LABELS))}")
    train = train.copy()
    train["LABEL"] = rounded.astype(int)
    return train


def read_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = root / "data" / "train.csv"
    test_path = root / "data" / "test.csv"
    sample_path = root / "data" / "sample_submission.csv"
    for path in [train_path, test_path, sample_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")
    train = pd.read_csv(train_path, dtype={"ID": "string", "TEXT": "string"})
    test = pd.read_csv(test_path, dtype={"ID": "string", "TEXT": "string"})
    sample = pd.read_csv(sample_path, dtype={"ID": "string"})
    require_columns(train, ["ID", "TEXT", "LABEL"], "train.csv")
    require_columns(test, ["ID", "TEXT"], "test.csv")
    require_columns(sample, ["ID", "LABEL"], "sample_submission.csv")
    validate_id_column(train, "train.csv")
    validate_id_column(test, "test.csv")
    validate_id_column(sample, "sample_submission.csv")
    train = validate_labels(train)
    if sample["ID"].astype("string").tolist() != test["ID"].astype("string").tolist():
        same_set = set(sample["ID"].astype("string")) == set(test["ID"].astype("string"))
        raise ValueError(f"sample_submission.csv IDs must match test.csv IDs in order. Same set: {same_set}.")
    return train, test, sample


def label_counts(series: pd.Series) -> dict[int, int]:
    return {label: int((series == label).sum()) for label in LABELS}


def duplicate_stats(keys: pd.Series) -> dict[str, int]:
    counts = keys.value_counts(dropna=False)
    duplicates = counts[counts > 1]
    return {
        "duplicate_group_count": int(len(duplicates)),
        "rows_in_duplicate_groups": int(duplicates.sum()) if not duplicates.empty else 0,
        "extra_duplicate_rows": int((duplicates - 1).sum()) if not duplicates.empty else 0,
    }


def add_duplicate_key(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    keyed["duplicate_key"] = keyed["TEXT"].map(make_duplicate_key)
    return keyed


def conflict_error_message(train_keyed: pd.DataFrame, keys: list[str], limit: int = 5) -> str:
    examples = []
    for key in keys[:limit]:
        rows = train_keyed.loc[train_keyed["duplicate_key"] == key, ["LABEL", "TEXT"]]
        labels = sorted(rows["LABEL"].unique().tolist())
        examples.append(f"key={key[:80]!r}, labels={labels}, rows={len(rows)}")
    return "Conflicting duplicate training-text labels found.\n" + "\n".join(examples)


# Remove leakage-like train/test text overlap and collapse exact duplicates.
def prepare_clean_views(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_keyed = add_duplicate_key(train)
    test_keyed = add_duplicate_key(test)
    unique_labels_by_key = train_keyed.groupby("duplicate_key", sort=False)["LABEL"].nunique()
    conflicts = unique_labels_by_key[unique_labels_by_key > 1].index.tolist()
    if conflicts:
        raise ValueError(conflict_error_message(train_keyed, conflicts))

    train_dups = duplicate_stats(train_keyed["duplicate_key"])
    test_dups = duplicate_stats(test_keyed["duplicate_key"])
    overlap_mask = train_keyed["duplicate_key"].isin(set(test_keyed["duplicate_key"]))
    train_after_overlap = train_keyed.loc[~overlap_mask].copy()
    train_clean = train_after_overlap.drop_duplicates("duplicate_key", keep="first").reset_index(drop=True)
    unique_test = test_keyed.drop_duplicates("duplicate_key", keep="first").reset_index(drop=True)
    summary = {
        "original_train_rows": int(len(train)),
        "original_test_rows": int(len(test)),
        "original_train_label_counts": label_counts(train["LABEL"]),
        "train_duplicate_groups": train_dups["duplicate_group_count"],
        "train_extra_duplicate_rows": train_dups["extra_duplicate_rows"],
        "test_duplicate_groups": test_dups["duplicate_group_count"],
        "test_extra_duplicate_rows": test_dups["extra_duplicate_rows"],
        "train_test_overlap_groups": int(train_keyed.loc[overlap_mask, "duplicate_key"].nunique()),
        "train_overlap_rows_removed": int(overlap_mask.sum()),
        "train_rows_after_overlap_removal": int(len(train_after_overlap)),
        "train_duplicate_rows_removed": int(len(train_after_overlap) - len(train_clean)),
        "clean_train_rows": int(len(train_clean)),
        "clean_train_label_counts": label_counts(train_clean["LABEL"]),
        "unique_test_rows_for_inference": int(len(unique_test)),
        "test_duplicate_rows_collapsed": int(len(test_keyed) - len(unique_test)),
        "conflicting_duplicate_label_groups": 0,
    }
    return train_clean, test_keyed, unique_test, summary


# Build word, character, and surface feature branches.
def build_feature_union(config: dict[str, Any]) -> FeatureUnion:
    negation = bool(config["negation"])
    parts = [
        ("word_tfidf", Pipeline([
            ("cleaner", CleanTextTransformer(negation)),
            ("tfidf", TfidfVectorizer(
                ngram_range=config["word"], min_df=WORD_MIN_DF, max_df=WORD_MAX_DF,
                sublinear_tf=True, dtype=np.float32,
            )),
        ])),
        ("char_wb_tfidf", Pipeline([
            ("cleaner", CleanTextTransformer(negation)),
            ("tfidf", TfidfVectorizer(
                analyzer=CHAR_ANALYZER, ngram_range=config["char"], min_df=config["char_min_df"],
                sublinear_tf=True, dtype=np.float32,
            )),
        ])),
    ]
    weights: dict[str, float] = {}
    if config["surface"] > 0.0:
        parts.append(("surface_features", Pipeline([
            ("surface", SurfaceFeaturesTransformer()),
            ("scaler", MaxAbsScaler()),
        ])))
        weights["surface_features"] = float(config["surface"])
    return FeatureUnion(transformer_list=parts, transformer_weights=weights or None)


def build_classifier(config: dict[str, Any]) -> SGDClassifier:
    return SGDClassifier(
        loss=config["loss"],
        penalty=SGD_SETTINGS["penalty"],
        alpha=float(config["alpha"]),
        class_weight=SGD_SETTINGS["class_weight"],
        max_iter=SGD_SETTINGS["max_iter"],
        tol=SGD_SETTINGS["tol"],
        n_iter_no_change=SGD_SETTINGS["n_iter_no_change"],
        random_state=RANDOM_SEED,
        average=SGD_SETTINGS["average"],
        n_jobs=SGD_SETTINGS["n_jobs"],
    )


def build_stage_pipeline(config: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("features", build_feature_union(config)),
        ("classifier", build_classifier(config)),
    ])


def evaluate_prediction(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).astype(int).tolist(),
    }



def train_validation_model(
    X_train: pd.Series,
    y_train: pd.Series,
    X_valid: pd.Series,
    y_valid: pd.Series,
) -> dict[str, Any]:
    model = HierarchicalSGDClassifier()
    model.fit(X_train, y_train)
    return evaluate_prediction(pd.Series(y_valid).astype(int), model.predict(X_valid))


def model_config() -> dict[str, Any]:
    return {
        "stage1": dict(STAGE1),
        "stage2": dict(STAGE2),
        "sgd": dict(SGD_SETTINGS),
        "surface_features": list(SURFACE_FEATURE_NAMES),
    }


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    labels = pd.to_numeric(submission["LABEL"], errors="coerce")
    integer_labels = bool(labels.notna().all() and np.allclose(labels, np.rint(labels)))
    checks = {
        "columns_exactly_ID_LABEL": list(submission.columns) == ["ID", "LABEL"],
        "row_count_matches_test": len(submission) == len(test),
        "id_order_matches_test": submission["ID"].astype("string").tolist() == test["ID"].astype("string").tolist(),
        "labels_are_allowed_integers": integer_labels and set(labels.astype(int).unique()).issubset(LABELS),
        "no_missing_predictions": int(labels.isna().sum()) == 0,
    }
    if not all(checks.values()):
        raise ValueError(f"Submission validation failed: {checks}")
    return checks


# Generate the submission and preserve original test ID order.
def make_submission(
    model: HierarchicalSGDClassifier,
    test_keyed: pd.DataFrame,
    unique_test: pd.DataFrame,
    path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    unique_predictions = model.predict(unique_test["TEXT"])
    prediction_by_key = dict(zip(unique_test["duplicate_key"], unique_predictions.astype(int)))
    expanded = test_keyed["duplicate_key"].map(prediction_by_key)
    if expanded.isna().any():
        raise ValueError(f"Missing predictions after duplicate expansion: {int(expanded.isna().sum())}")
    submission = pd.DataFrame({"ID": test_keyed["ID"].astype("string"), "LABEL": expanded.astype(int)})
    checks = validate_submission(submission, test_keyed)
    submission.to_csv(path, index=False)
    return submission, checks


def report_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2e}" if value != 0.0 and abs(value) < 0.001 else f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(report_value(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def label_count_rows(counts: dict[int, int]) -> list[list[Any]]:
    total = sum(counts.values())
    return [
        [
            label,
            LABEL_MEANINGS[label],
            counts.get(label, 0),
            100 * counts.get(label, 0) / total if total else 0.0,
        ]
        for label in LABELS
    ]


def per_class_rows(report: dict[str, Any]) -> list[list[Any]]:
    return [[
        label, LABEL_MEANINGS[label], float(report[str(label)]["precision"]),
        float(report[str(label)]["recall"]), float(report[str(label)]["f1-score"]),
        int(report[str(label)]["support"]),
    ] for label in LABELS]


def confusion_matrix_rows(confusion: list[list[int]]) -> list[list[Any]]:
    return [[f"actual {label}"] + [int(value) for value in row] for label, row in zip(LABELS, confusion)]


def tuple_str(value: tuple[int, int]) -> str:
    return f"({value[0]}, {value[1]})"


def stage_rows(prefix: str, config: dict[str, Any]) -> list[list[Any]]:
    return [
        [f"{prefix}_task", config["task"]],
        [f"{prefix}_word_ngram_range", tuple_str(config["word"])],
        [f"{prefix}_char_ngram_range", tuple_str(config["char"])],
        [f"{prefix}_char_min_df", config["char_min_df"]],
        [f"{prefix}_loss", config["loss"]],
        [f"{prefix}_alpha", config["alpha"]],
        [f"{prefix}_surface_weight", config["surface"]],
        [f"{prefix}_negation_scope", config["negation"]],
    ]


# Write a compact report with configuration, validation metrics, and checks.
def render_report(
    root: Path,
    paths: dict[str, Path],
    cleaning: dict[str, Any],
    result: dict[str, Any],
    submission_checks: dict[str, Any],
    submission_rows: int,
) -> str:
    config_rows = [
        ["model_name", MODEL_NAME],
        ["random_seed", RANDOM_SEED],
        ["validation_strategy", f"stratified holdout, test_size={VALIDATION_SIZE}"],
        ["selection_metric", "macro F1"],
        ["word_min_df", WORD_MIN_DF],
        ["word_max_df", WORD_MAX_DF],
        ["char_analyzer", CHAR_ANALYZER],
    ] + stage_rows("stage1", STAGE1) + stage_rows("stage2", STAGE2)
    cleaning_rows = [[k, v] for k, v in cleaning.items() if "counts" not in k]
    check_rows = [[k, v] for k, v in submission_checks.items()]
    check_rows.append(["submission_row_count", submission_rows])
    lines = [
        "# Model Results", "", "Generated by `scripts/train.py`.", "",
        "## Output files", "",
        markdown_table(["artifact", "path"], [
            ["model artifact", relative_path(paths["model"], root)],
            ["submission CSV", relative_path(paths["submission"], root)],
            ["model results report", relative_path(paths["report"], root)],
        ]), "",
        "## Data cleaning summary", "",
        markdown_table(["cleaning statistic", "value"], cleaning_rows), "",
        "### Original label counts", "",
        markdown_table(
            ["LABEL", "meaning", "count", "percent"],
            label_count_rows(cleaning["original_train_label_counts"]),
        ), "",
        "### Clean training label counts", "",
        markdown_table(
            ["LABEL", "meaning", "count", "percent"],
            label_count_rows(cleaning["clean_train_label_counts"]),
        ), "",
        "## Selected model configuration", "",
        markdown_table(["setting", "value"], config_rows), "",
        "## Validation metrics", "",
        markdown_table(["metric", "value"], [
            ["validation_macro_f1", result["macro_f1"]],
            ["validation_accuracy", result["accuracy"]],
        ]), "",
        "## Per-class validation metrics", "",
        markdown_table(
            ["LABEL", "meaning", "precision", "recall", "f1", "support"],
            per_class_rows(result["classification_report"]),
        ), "",
        "## Validation confusion matrix", "",
        markdown_table(
            ["actual/predicted", "predicted 0", "predicted 1", "predicted 2"],
            confusion_matrix_rows(result["confusion_matrix"]),
        ), "",
        "## Submission validation", "", markdown_table(["check", "value"], check_rows), "",
    ]
    return "\n".join(lines)


# Store only metadata needed to reproduce and identify the model artifact.
def build_metadata(
    root: Path,
    paths: dict[str, Path],
    cleaning: dict[str, Any],
    result: dict[str, Any],
    submission_checks: dict[str, Any],
) -> dict[str, Any]:
    report = result["classification_report"]
    return {
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "label_mapping": LABEL_MEANINGS,
        "validation_strategy": {
            "type": "stratified_holdout",
            "test_size": VALIDATION_SIZE,
            "random_seed": RANDOM_SEED,
        },
        "validation_macro_f1": result["macro_f1"],
        "validation_accuracy": result["accuracy"],
        "validation_per_class_f1": {label: float(report[str(label)]["f1-score"]) for label in LABELS},
        "selected_config": model_config(),
        "cleaning_summary": cleaning,
        "submission_validation": submission_checks,
        "paths": {name: relative_path(path, root) for name, path in paths.items()},
    }


# Run training, validation, artifact saving, and submission generation.
def main() -> None:
    np.random.seed(RANDOM_SEED)
    root = PROJECT_ROOT
    models_dir = root / "models"
    submissions_dir = root / "submissions"
    reports_dir = root / "reports"
    for output_dir in [models_dir, submissions_dir, reports_dir]:
        output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": models_dir / f"{MODEL_NAME}.joblib",
        "submission": submissions_dir / f"submission_{MODEL_NAME}.csv",
        "report": reports_dir / "model_results.md",
    }

    print("Loading data...")
    train, test, _sample = read_inputs(root)
    print("Applying duplicate and overlap cleanup...")
    train_clean, test_keyed, unique_test, cleaning = prepare_clean_views(train, test)

    print("Creating validation split...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_clean["TEXT"], train_clean["LABEL"].astype(int), test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED, stratify=train_clean["LABEL"].astype(int),
    )

    print("Training fixed selected model for validation...")
    validation_result = train_validation_model(X_train, y_train, X_valid, y_valid)
    print(f"Validation macro F1: {validation_result['macro_f1']:.4f}")

    print("Fitting final model on the full cleaned training view...")
    final_model = HierarchicalSGDClassifier()
    final_model.fit(train_clean["TEXT"], train_clean["LABEL"].astype(int))

    print("Generating submission...")
    submission, submission_checks = make_submission(final_model, test_keyed, unique_test, paths["submission"])

    print("Saving model artifact...")
    metadata = build_metadata(root, paths, cleaning, validation_result, submission_checks)
    joblib.dump({"model": final_model, "metadata": metadata}, paths["model"], compress=3)

    print("Writing report...")
    report_text = render_report(root, paths, cleaning, validation_result, submission_checks, len(submission))
    paths["report"].write_text(report_text, encoding="utf-8")

    print("Done.")
    print(f"- {relative_path(paths['model'], root)}")
    print(f"- {relative_path(paths['submission'], root)}")
    print(f"- {relative_path(paths['report'], root)}")


if __name__ == "__main__":
    main()
