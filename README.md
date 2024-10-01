#Airflow and MLflow Integration Example

## Description
The approach of this project is facilitates the orchestration and tracking of machine learning workflows through MLflow.
	-MLflow was used for the experiment tracking and organization.
	-Airflow for the run orchestration between the stages of MLflow.
![[pics/Untitled Diagram.drawio.png]]

We found inspiration in this [Astronomer.io tutorial](https://www.astronomer.io/docs/learn/airflow-mlflow)
The workflow is shown below
![[pics/workflow.png]]
The flow is integrated by 4 main tasks:
	-  create_buckets_if_not_exists
	-  prepare_mlflow_experiment
	-  train_model
	-  create_registered_model
We are generated the module named include/model.py that contains the Model Class. This class contains information like:

```python 
class Model:
	def __init__(self):
		self.ml_flow_artifact_bucket = "mlflowdataredwine"
		self.experiment_name = "wine_quality"
		self.file_path = "include/winequalityred.csv"
		self.alpha = 0.7
		self.l1_ratio =0.7
```
With this attributes we can specify the name of experiment, the bucket the artifacts of experiment, the data path and the hyper parameters for the model.

Additionaly we can set the experiment and instrumented with MLflow and/ or evidently.ai


```python
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
```

## How to use this repository

This section explains how to run this repository with Airflow. Note that you will need to copy the contents of the `.env_example` file to a newly created `.env` file. No external connections are necessary to run this repository locally, but you can add your own credentials in the file if you wish to connect to your tools. 

### Option 1: Use GitHub Codespaces

Run this Airflow project without installing anything locally.

1. Fork this repository.
2. Create a new GitHub codespaces project on your fork. Make sure it uses at least 4 cores!
3. After creating the codespaces project the Astro CLI will automatically start up all necessary Airflow components and the local MinIO and MLflow instances. This can take a few minutes. 
4. Once the Airflow project has started, access the Airflow UI by clicking on the **Ports** tab and opening the forward URL for port 8080. The MLflow instance is accessible at port 5000, the MinIO instance at port 9000.

### Option 2: Use the Astro CLI

Download the [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli) to run Airflow locally in Docker. `astro` is the only package you will need to install locally.

1. Run `git clone https://github.com/astronomer/use-case-mlflow.git` on your computer to create a local clone of this repository.
2. Install the Astro CLI by following the steps in the [Astro CLI documentation](https://docs.astronomer.io/astro/cli/install-cli). Docker Desktop/Docker Engine is a prerequisite, but you don't need in-depth Docker knowledge to run Airflow with the Astro CLI.
3. Run `astro dev start` in your cloned repository.
4. After your Astro project has started. View the Airflow UI at `localhost:8080`, the MLflow UI at `localhost:5000` and the MinIO UI at `localhost:9000`.


#Create a new project or update the model

All modification related to training model (change of ML algorithm, change of data inputs, etc) can be applied directly on the module model.py located in include folder. 
Example: Change the values for Alpha and L1-ratio

Changing Alpha and L1-ration on model.py
![[pics/changing_parameters.png]]

Run the dag wine_feature_eng
![[pics/airflow_running.png]]

When the execution is completed, you can go to MLflow UI and check the new run for the experiment
![[pics/mlflow_tracking.png]]

And verify the model registered
![[pics/register.png]]
