# Using MLOps Tools

(You can work in pairs if you want to.)

This is the base repository you should use for your own GitHub repository (not a TIK Enterprise GitHub, but [Microsoft's public GitHub instance](https://github.com/)).

1. Copy these files into the local directory of your own GitHub repo.
2. Install the dependencies (`pip install -r requirements.txt`).
3. Import the needed dataset with DVC.
4. Extend the script (`src/train.py`) with the necessary MLflow experiment tracking and model storing code (see `TODO` comments).
5. Run the script (`python src/train.py`) and look at the results in the MLflow UI.
6. Create a CI pipeline with GitHub Actions.