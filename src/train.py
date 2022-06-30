# adapted from https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_wine/train.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    # Root Mean Square Error (https://en.wikipedia.org/wiki/Root-mean-square_deviation)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    # Mean Absolute Error (https://en.wikipedia.org/wiki/Mean_absolute_error)
    mae = mean_absolute_error(actual, pred)
    # R squared / Coefficient of Determination (https://en.wikipedia.org/wiki/Coefficient_of_determination)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# set fixed random seed for numpy for reproducibility
np.random.seed(40)

# read the csv file from the local directory (needs to have been imported / pulled with DVC)
try:
    data = pd.read_csv("winequality-red.csv", sep=";")
except Exception as e:
    logger.exception(
        "Unable to find CSV file! Have you imported it with DVC? Error: %s", e
    )

# split the data into training and test sets (0.75 and 0.25)
train, test = train_test_split(data)

# the column to be predicted is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# start training loop to test 10 different hyperparameter settings
for x in range(10):

    # set hyperparameters based on loop variable (0.01 to 0.91)
    alpha = 0.01 + x / 10
    l1_ratio = 0.01 + x / 10

    with mlflow.start_run():
        # instantiate the model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        # train the model with the training data
        lr.fit(train_x, train_y)
        # evaluate the model on the test data
        predicted_qualities = lr.predict(test_x)
        # calculate evaluation metrics based on prediction results
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Training loop #%s" % (x + 1))
        print("  Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("    RMSE: %s" % rmse)
        print("    MAE: %s" % mae)
        print("    R2: %s" % r2)

        # log experiment params and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # store the model
        mlflow.sklearn.log_model(lr, "model")
