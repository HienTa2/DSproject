import os
import sys
import json
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src_scripts.exception import CustomException
from src_scripts.logger import logging
from src_scripts.utils import save_object, evaluate_models


class CustomXGBRegressor(XGBRegressor):
    def __sklearn_tags__(self):
        return {"estimator_type": "regressor"}


@dataclass
class ModelTrainerConfig:
    base_artifacts_path: str = os.path.join("artifacts")
    trained_model_file_path: str = os.path.join(base_artifacts_path, "models", "model.pkl")


class ModelTrainer:
    def __init__(self, config_path="config.json"):
        # Dynamically resolve the full path to config.json
        self.config_path = os.path.abspath(config_path)
        self.model_trainer_config = ModelTrainerConfig()

    def get_models_and_params(self):
        logging.info(f"Loading configuration from: {self.config_path}")
        try:
            # Load the configuration file
            with open(self.config_path, "r") as file:
                config = json.load(file)
        except FileNotFoundError:
            raise CustomException(f"Configuration file not found: {self.config_path}", sys)
        except json.JSONDecodeError as e:
            raise CustomException(f"Invalid JSON format in config file: {self.config_path}. Error: {e}", sys)

        # Map model names to their respective classes
        model_mapping = {
            "RandomForestRegressor": RandomForestRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "LinearRegression": LinearRegression,
            "CustomXGBRegressor": CustomXGBRegressor,
            "CatBoostRegressor": lambda: CatBoostRegressor(verbose=False),
            "AdaBoostRegressor": AdaBoostRegressor,
        }

        # Validate models in the configuration
        unsupported_models = [name for name in config["models"] if name not in model_mapping]
        if unsupported_models:
            raise CustomException(f"Unsupported models in config: {unsupported_models}", sys)

        # Instantiate models and retrieve parameters
        models = {name: model_mapping[name]() for name in config["models"]}
        params = config.get("models_params", {})
        return models, params

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            # Split data into features and labels
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Retrieve models and their parameters
            models, params = self.get_models_and_params()

            # Evaluate all models
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params,
            )

            # Filter valid model scores
            valid_model_scores = {name: score for name, score in model_report.items() if score is not None}
            if not valid_model_scores:
                raise CustomException("No valid model scores available. Check model evaluation and parameters.")

            # Determine the best model
            best_model_score = max(valid_model_scores.values())
            best_model_name = max(valid_model_scores, key=valid_model_scores.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² > 0.6")

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"R² Score: {best_model_score}")

            # Ensure model directory exists
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            # Test the best model on the test set
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return {
                "Best Model": best_model_name,
                "R² Score": r2_square,
                "Explanation": "R² measures how well the model explains the variance in the data. "
                               "A score closer to 1 means better performance.",
            }

        except Exception as e:
            raise CustomException(e, sys)
