import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, RegressionPreset
from evidently.metrics import *



import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self):
        self.ml_flow_artifact_bucket = "mlflowdataredwine"
        self.experiment_name = "wine_quality"
        self.classification_type= "svm"
        
        self.file_path = "include/data/winequalityred.csv"
        
    
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
        data_drift_suite = TestSuite(tests=[DataDriftTestPreset()])
        data_drift_suite.run(reference_data=data, current_data=train)
        


        classifier = SVC(random_state = 0, kernel = 'linear')
        classifier.fit(train_x, train_y)
        predicted_qualities = classifier.predict(test_x)

        data_drift_suite = TestSuite(tests=[DataDriftTestPreset()])
        data_drift_suite.run(reference_data=data, current_data=train)
        

        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
        mlflow.log_param("kernel", 'linear')
        mlflow.log_metric("rmse", rmse) 
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_artifact(self.file_path)
        data_drift_suite.save_html("data_drift_suite.html")
        mlflow.log_artifact('data_drift_suite.html')
        mlflow.sklearn.log_model(classifier, "svm_linear")