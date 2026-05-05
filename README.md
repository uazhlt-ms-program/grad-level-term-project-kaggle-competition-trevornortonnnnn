# Movie and TV Review Text Classification

This repository contains a complete text classification workflow for assigning each input document to one of three labels:

| Label | Meaning |
| ---: | --- |
| `0` | Not a movie or TV show review |
| `1` | Positive movie or TV show review |
| `2` | Negative movie or TV show review |

The classifier handles two related decisions: first, whether a text is a movie/TV review at all, and second, whether a review is positive or negative.

## Repository Structure

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/
│   ├── .gitkeep
│   └── tfidf_hierarchical_sgd_hinge_refined_v2.joblib
├── reports/
│   ├── .gitkeep
│   ├── model_results.md
│   └── stat_report.md
├── scripts/
│   ├── explore_data.py
│   └── train.py
├── submissions/
│   ├── .gitkeep
│   └── submission_tfidf_hierarchical_sgd_hinge_refined_v2.csv
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Data Requirements and Schema

The workflow expects the project data in `data/`.

| File | Required columns | Purpose |
| --- | --- | --- |
| `data/train.csv` | `ID`, `TEXT`, `LABEL` | Labeled training data |
| `data/test.csv` | `ID`, `TEXT` | Unlabeled test data for prediction |
| `data/sample_submission.csv` | `ID`, `LABEL` | Submission template and expected ID order |

The scripts validate the expected columns and ID structure before running. Raw CSV files in `data/` are treated as read-only inputs.

## Modeling Approach

The final model is implemented in `scripts/train.py` as a custom scikit-learn estimator named `HierarchicalSGDClassifier`.

The model uses a two-stage hierarchy:

1. **Stage 1: review detection**
   - Binary task: label `0` versus labels `1` and `2`
   - Texts predicted as non-reviews are assigned label `0`

2. **Stage 2: review sentiment**
   - Binary task: label `1` versus label `2`
   - Only texts predicted as reviews by Stage 1 are passed to this classifier

Both stages use linear `SGDClassifier` models with hinge loss, L2 regularization, averaged weights, and TF-IDF-based feature pipelines. The final trained model is saved as:

```text
models/tfidf_hierarchical_sgd_hinge_refined_v2.joblib
```

## Feature Extraction and Text Normalization

Each classifier stage uses a `FeatureUnion` that combines three feature branches:

| Feature branch | Description |
| --- | --- |
| Word TF-IDF | Word-level n-gram features extracted from normalized text |
| Character TF-IDF | Character-level `char_wb` n-gram features for subword and spelling patterns |
| Surface features | Numeric features based on text length, punctuation, casing, digits, and repeated punctuation |

The text cleaning pipeline performs the following normalization steps:

- Converts missing text values to empty strings
- Unescapes HTML entities
- Replaces `<br>` tags and strips other HTML tags
- Applies Unicode normalization
- Normalizes quotes, apostrophes, dashes, ellipses, and non-breaking spaces
- Removes invisible and control characters
- Replaces email addresses with `__email__`
- Replaces URLs with `__url__`
- Expands common contractions
- Normalizes ratings such as `8/10`, `8 out of 10`, `4 stars`, and selected letter grades into rating tokens
- Converts repeated or mixed punctuation into normalized punctuation tokens
- Shortens elongated character sequences
- Collapses repeated whitespace

The surface feature branch extracts:

- Log character count
- Log word count
- Sentence punctuation count
- Exclamation mark count
- Question mark count
- Uppercase letter ratio
- Digit ratio
- Punctuation ratio
- Repeated punctuation flag

Surface features are scaled with `MaxAbsScaler`.

Stage 2 also applies negation-scope marking so that words following negation triggers such as `not`, `never`, or `without` receive a `_NEG` suffix within a short window.

## Data Cleaning in the Training Workflow

The training script creates normalized duplicate keys, removes exact train/test text overlap from the training view, collapses duplicate training text, deduplicates test text for inference, and expands predictions back to the original test ID order.

| Cleaning statistic | Count |
| --- | ---: |
| Original training rows | `70,317` |
| Original test rows | `17,580` |
| Training duplicate groups | `362` |
| Extra training duplicate rows | `1,151` |
| Test duplicate groups | `71` |
| Extra test duplicate rows | `181` |
| Train/test overlap groups | `209` |
| Training overlap rows removed | `1,003` |
| Training duplicate rows removed after overlap removal | `357` |
| Clean training rows used for modeling | `68,957` |
| Unique test rows used for inference | `17,399` |
| Test duplicate rows collapsed for inference | `181` |
| Conflicting duplicate-label groups | `0` |

## Reproducing the Workflow

Docker is the recommended way to run the project.

### 1. Build the Docker image

From the repository root:

```bash
docker build -t classification-project:py313 .
```

### 2. Enter the container

```bash
docker run --rm -it \
  --mount "type=bind,source=${PWD},target=/app" \
  -w /app \
  classification-project:py313 \
  bash
```

### 3. Run the training workflow inside the container

Once inside the container, run:

```bash
python scripts/train.py
```

This writes or updates:

```text
reports/model_results.md
models/tfidf_hierarchical_sgd_hinge_refined_v2.joblib
submissions/submission_tfidf_hierarchical_sgd_hinge_refined_v2.csv
```

Exit the container when finished:

```bash
exit
```

## Direct Docker Commands

The same workflow can also be run without manually entering the container.

Run data exploration:

```bash
docker run --rm \
  --mount "type=bind,source=${PWD},target=/app" \
  -w /app \
  classification-project:py313 \
  python scripts/explore_data.py
```

Run training and submission generation:

```bash
docker run --rm \
  --mount "type=bind,source=${PWD},target=/app" \
  -w /app \
  classification-project:py313 \
  python scripts/train.py
```