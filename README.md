# Using MLOps Tools

(You can work in pairs if you want to.)

In this tutorial, you are going to use DVC and MLflow to set up an existing ML pipeline in a [GitHub repository](https://github.com), look at the experiment results, and finally build a CI Pipeline with [GitHub Actions](https://docs.github.com/en/actions/quickstart).
This is the starter repository you should use as the basis for your own GitHub repository.

**Prerequisites:**
- Install [DVC](https://dvc.org)
- Create a [GitHub account](https://github.com/signup) (if you don't have one already)
- Create a [new GitHub repository](https://github.com/new) for your account and initialize the repo with a `README` to allow direct cloning (alternative: fork this starter repo)
- Clone the new or forked repo to your local machine
- Install the dependencies (`pip install -r requirements.txt`), which will also install [MLflow](https://mlflow.org/docs/latest/quickstart.html#installing-mlflow)

## Part 1: Using DVC and MLflow

Before you start working on the tasks below, it might be a good idea to first check out the quickstart guides for both [DVC](https://dvc.org/doc/start) and [MLflow](https://mlflow.org/docs/latest/quickstart.html).

**Tasks:**
1. Copy the files from the [starter repo](https://github.com/se4ai-lecture/mlops-tools) to your local repo (not needed if you forked it).
2. Initialize DVC in the local repo.
3. The training script (`src/train.py`) requires a CSV dataset for the quality of red wine. Use the `dvc list` command to display all DVC artifacts in the `data` folder of the following repo: https://github.com/se4ai-lecture/dvc-artifacts
4. After you've identified the correct CSV file, use `dvc import` to import the remote dataset into your own repo.
5. Extend the script (`src/train.py`) with the necessary MLflow experiment tracking and model storing code (see `TODO` comments). Two hyperparameters and three evaluation metrics should be tracked.
6. Execute the updated training script to start the experiments (`python src/train.py`).
7. Start the MLflow web UI and look at the experiment results. Which hyperparameter configuration leads to the best R2 score?
8. Commit and push everything to your GitHub repo (except for the `mlruns` folder and the actual dataset, but those are already excluded via `.gitignore`).

## Part 2: Creating a CI Pipeline with GitHub Actions

Before you start working on the task below, it might be a good idea to first check out the quickstart guide for [GitHub Actions](https://docs.github.com/en/actions/quickstart).

Setup a CI pipeline using GitHub Actions to automatically run the training script created in part 1 on every push.
The [Python Starter Workflow](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) might be a good basis that you can adapt.
Remember that the CI runner will also need to retrieve the dataset via DVC.