# Text Classification Project

This repository contains the source code, data, environment setup, and project workflow for the classification project.

The project is a three-label text classification task. Each document is classified as one of the following labels:

| Label | Meaning |
|---:|---|
| `0` | Not a movie or TV show review |
| `1` | Positive movie or TV show review |
| `2` | Negative movie or TV show review |

The project is evaluated using macro F1, so the goal is to build a classifier that performs reasonably well across all three labels instead of only optimizing overall accuracy.

## Project goals

The main goals of this project are:

1. Explore the project data.
2. Build a reproducible text-classification workflow.
3. Train and evaluate baseline models.
4. Generate project submission files.
5. Document the workflow clearly enough that the results can be reproduced from this repository.

The repository is organized so that the full workflow can be run through Docker.

## Current project status

The project is currently in the setup and reproducibility stage.

Completed so far:

- The repository has been cloned and opened in VS Code.
- The project data has been added under `data/`.
- The Python dependency list has been updated for Python 3.13.
- The project is being moved into a Docker-based workflow.
- Initial project scripts are being organized under `scripts/`.

Next steps:

- Finish the Docker reproducibility setup.
- Run the first exploratory data analysis script.
- Build a baseline text-classification model.
- Generate the first submission file.
- Add validation results and error analysis.

## Repository structure

```text
.
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
├── scripts/
│   ├── check_environment.py
│   └── explore_data.py
├── submissions/
│   └── .gitkeep
├── Dockerfile
├── README.md
├── requirements.txt
├── .dockerignore
└── .gitignore

```

## Data

The data files for the project are in the `data/` directory.

Expected files:

- `data/train.csv`
- `data/test.csv`
- `data/sample_submission.csv`

These files are committed to the repository so that the project can be reproduced without a separate data-download step.

The raw data files should not be edited manually. Scripts should read from `data/` and write generated outputs to `reports/`, `models/`, or `submissions/`.

## Environment

The project uses Docker to create a reproducible Python 3.13 environment.

The Docker image installs the packages listed in `requirements.txt` and runs project scripts from the repository root.

## Build the Docker image

From the repository root, run:

```bash
docker build -t classification-project:py313 .
```

## Check the Docker environment

Run:

```bash
docker run --rm classification-project:py313
```

The default Docker command runs:

```bash
python scripts/check_environment.py
```

This verifies the Python version, installed packages, and expected project folders.

To run the environment check while mounting the local repository, use:

```bash
docker run --rm -it --mount "type=bind,source=${PWD},target=/app" -w /app classification-project:py313 python scripts/check_environment.py
```

## Current scripts

### `scripts/check_environment.py`

Checks that the Docker environment is usable. It prints the Python version, platform, package versions, and expected project paths.

### `scripts/explore_data.py`

Exploratory data-analysis script. This script is used to inspect the training and test data, summarize label counts, check missing values, examine text lengths, and generate useful outputs for the final project write-up.

## Planned modeling workflow

The planned workflow is:

1. Load `train.csv` and `test.csv`.
2. Split the training data into training and validation sets.
3. Train baseline text classifiers using TF-IDF features.
4. Evaluate models using macro F1.
5. Analyze validation errors.
6. Train the final selected model.
7. Generate a submission CSV in `submissions/`.

Planned scripts include:

```text
scripts/train_baseline.py
scripts/evaluate_model.py
scripts/make_submission.py
```

## Output folders

### `reports/`

Stores generated EDA summaries, validation results, plots, and error-analysis notes.

### `models/`

Stores trained model artifacts.

### `submissions/`

Stores generated submission CSV files.

## Reproducibility notes

The project is designed so that a reader can clone the repository, build the Docker image, run the scripts, and reproduce the project workflow.

As the project develops, this README will be updated with:

- EDA commands
- training commands
- validation results
- submission-generation commands
- final model information
- final project results