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