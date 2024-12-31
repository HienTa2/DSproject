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
    trained_model_file_path = os.path.join("artifacts", "models", "model.pkl")


class ModelTrainer:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.model_trainer_config = ModelTrainerConfig()

    def get_models_and_params(self):
        try:
            with open(self.config_path, "r") as file:
                config = json.load(file)
        except FileNotFoundError:
            raise CustomException(f"Configuration file not found: {self.config_path}", sys)
        except json.JSONDecodeError:
            raise CustomException(f"Invalid JSON format in config file: {self.config_path}", sys)

        # Explicit mapping for model initialization
        model_mapping = {
            "RandomForestRegressor": RandomForestRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "LinearRegression": LinearRegression,
            "CustomXGBRegressor": CustomXGBRegressor,
            "CatBoostRegressor": lambda: CatBoostRegressor(verbose=False),
            "AdaBoostRegressor": AdaBoostRegressor
        }

        unsupported_models = [name for name in config["models"] if name not in model_mapping]
        if unsupported_models:
            raise CustomException(f"Unsupported models in config: {unsupported_models}", sys)

        models = {name: model_mapping[name]() for name in config["models"]}
        params = config.get("models_params", {})
        return models, params

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Retrieve models and parameters
            models, params = self.get_models_and_params()

            # Evaluate models
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Get the best model and its score
            valid_model_scores = {name: score for name, score in model_report.items() if score is not None}

            if not valid_model_scores:
                raise CustomException("No valid model scores available. Check model evaluation and parameters.")

            best_model_score = max(valid_model_scores.values())
            best_model_name = max(valid_model_scores, key=valid_model_scores.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² > 0.6")

            logging.info(f"Best Model: {best_model_name}")
            logging.info(
                f"R² Score: {best_model_score} (R² measures how well the model explains the variance in the data. "
                "A score closer to 1 means better performance.)")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Make predictions and compute final R² score
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return {
                "Best Model": best_model_name,
                "R² Score": r2_square,
                "Explanation": "R² measures how well the model explains the variance in the data. "
                               "A score closer to 1 means better performance."
            }

        except Exception as e:
            raise CustomException(e, sys)


