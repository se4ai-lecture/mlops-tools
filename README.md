# Using MLOps Tools

(You can work in pairs if you want to.)

This is the base repository you should use for your own GitHub repository (not a TIK Enterprise GitHub, but [Microsoft's public GitHub instance](https://github.com/)).

1. Copy these files into the local directory of your own GitHub repo.
2. Install the dependencies (`pip install -r requirements.txt`).
3. Import the needed dataset with DVC.
4. Extend the script (`src/train.py`) with the necessary MLflow experiment tracking and model storing code (see `TODO` comments).
5. Run the script (`python src/train.py`) and look at the results in the MLflow UI.
6. Create a CI pipeline with GitHub Actions.

Necessary steps to arrive at the solution:
```bash
# copy requirements.txt and install the Python dependencies
pip install -r requirements.txt

# initialize DVC in the git repo
dvc init

# list all DVC artifacts in the data folder of the named repo (3 in total)
dvc list https://github.com/xJREB/se4ai-lecture-dvc-artifacts data

# import the correct data set from the repo (if you clone this repo, `dvc pull` is enough)
dvc import https://github.com/xJREB/se4ai-lecture-dvc-artifacts data/winequality-red.csv

# copy and run the extended training script
python src/train.py

# open the MLflow web interface --> sort the experiment table by r2 score and look which alpha and l1_ratio the highest value has (r2 of 0.358 for alpha=0.01 and l1_ratio=0.01)
mlflow ui

# lastly, build a CI pipeline with GitHub actions (see `.github/workflows/ml-pipeline.yml`)
```