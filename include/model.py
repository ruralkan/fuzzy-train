import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self):
        self.ml_flow_artifact_bucket = "mlflowdataredwine"
        self.experiment_name = "wine_quality"
        self.file_path = "include/winequalityred.csv"
        self.alpha = 0.7
        self.l1_ratio =0.7
    
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def run_model(self):
        warnings.filterwarnings("ignore")
        np.random.seed(40)
        # Read the wine-quality csv file from local
        data = pd.read_csv(self.file_path)
        data.to_csv(self.file_path, index=False)
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)
        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        alpha = self.alpha
        l1_ratio = self.l1_ratio

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_artifact(self.file_path)
        mlflow.sklearn.log_model(lr, "wine_regression")