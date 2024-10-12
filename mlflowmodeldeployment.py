import os
from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
from typing import List


MLFLOW_TRACKING_URI = 'https://localhost:5000' #os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = "wine_quality" # Defined previously by you
MLFLOW_MODEL_NAME = "wine_quality"

CHAMPION_MODEL_ALIAS = "champion"

class ModelManager:
    """Model Manager for AUTOMATIC   Deployment"""

    def __init__(
        self,
        mlflow_tracking_uri: str,
        mlflow_experiment_name: str,
        mlflow_model_name: str,
    ):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_model_name = mlflow_model_name
        self.client = MlflowClient()

    def search_best_model(
        self, experiment_names: List[str] = [], metric_name: str = "r2"
    ) -> str:
        """Search Best Run ID of given experiments"""
        logger.info("Searching best model...")
        logger.info(f"this is MLFLOW_TRACKING_URI {MLFLOW_TRACKING_URI}")
        runs_ = mlflow.search_runs(experiment_names=experiment_names)
        best_run = runs_.loc[runs_[f"metrics.{metric_name}"].idxmax()]

        return best_run["run_id"], f"{best_run['artifact_uri']}/model"

    def promote_model(self, run_id: str, artifact_uri: str, model_name: str) -> ModelVersion:
        """Promote a model to a new alias"""
        return mlflow.register_model(
            model_uri=f"runs:/{run_id}/{artifact_uri}", name=model_name
        )

    def run_deploy(
        self,
        run_id: str,
        artifact_uri: str,
        model_name: str,
    ) -> None:
        """Deploy a model to a new alias"""
        _new_model = self.promote_model(run_id, artifact_uri, model_name)
        print(_new_model)
        if _new_model.version == "1":
            logger.info("First model version, setting as champion model.")
            self.client.set_registered_model_alias(
                MLFLOW_MODEL_NAME, CHAMPION_MODEL_ALIAS, _new_model.version
            )
        else:
            logger.info(
                f"New model version: v{_new_model.version}, Verifying if the model is different from current champion."
            )
            champ_model = self.client.get_model_version_by_alias(
                MLFLOW_MODEL_NAME, CHAMPION_MODEL_ALIAS
            )
            if best_run_id == champ_model.run_id:
                logger.info(
                    "Best model found is already champion model, no need to update. Exiting."
                )
            else:
                logger.info(
                    """Best model is not champion model, Promoting new model.
                    """
                )
                self.client.set_registered_model_alias(
                    MLFLOW_MODEL_NAME, CHAMPION_MODEL_ALIAS, _new_model.version
                )

if __name__ == "__main__":
    logger.info("Starting Automatic Model Deployment...")
    logger.info(f"This is MLFLOW_TRACKING_URI in main {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(
            uri=MLFLOW_TRACKING_URI,
        )
    mlflow.set_experiment(experiment_name=MLFLOW_MODEL_NAME)
    manager = ModelManager(
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        mlflow_experiment_name=MLFLOW_EXPERIMENT,
        mlflow_model_name=MLFLOW_MODEL_NAME,
    )

    best_run_id, best_run_art_uri = manager.search_best_model(
        experiment_names=[MLFLOW_EXPERIMENT]
    )

    manager.run_deploy(
        run_id=best_run_id,
        artifact_uri=best_run_art_uri,
        model_name=MLFLOW_MODEL_NAME,
    )

    logger.info("Automatic Deployment applied successfully.")