from xgboost import XGBRegressor
import os
import sys
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src_scripts.exception import CustomException

# Suppress warnings about __sklearn_tags__
import warnings
warnings.filterwarnings("ignore", message=".*__sklearn_tags__.*")

# Custom wrapper for XGBRegressor to fix compatibility
class CustomXGBRegressor(XGBRegressor):
    def __sklearn_tags__(self):
        return {"estimator_type": "regressor"}

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using manual tuning for XGBRegressor and GridSearchCV for others.

    Args:
        X_train (array): Training features.
        y_train (array): Training target.
        X_test (array): Testing features.
        y_test (array): Testing target.
        models (dict): Dictionary of models to evaluate.
        param (dict): Dictionary of hyperparameters for each model.

    Returns:
        dict: R² scores for each model after tuning.
    """
    try:
        report = {}

        for model_name, model in models.items():
            if model_name == "CustomXGBRegressor":
                # Manual tuning for CustomXGBRegressor
                best_params = None
                best_score = float("-inf")

                for learning_rate in param[model_name]["learning_rate"]:
                    for n_estimators in param[model_name]["n_estimators"]:
                        model.set_params(learning_rate=learning_rate, n_estimators=n_estimators)
                        model.fit(X_train, y_train)

                        y_test_pred = model.predict(X_test)
                        score = r2_score(y_test, y_test_pred)

                        if score > best_score:
                            best_score = score
                            best_params = {"learning_rate": learning_rate, "n_estimators": n_estimators}

                model.set_params(**best_params)  # Apply best parameters
                report[model_name] = best_score

            else:
                # Use GridSearchCV for other models
                try:
                    gs = GridSearchCV(model, param[model_name], cv=3, scoring='r2', n_jobs=-1)
                    gs.fit(X_train, y_train)

                    # Update model with best parameters
                    model.set_params(**gs.best_params_)
                    model.fit(X_train, y_train)

                    y_test_pred = model.predict(X_test)
                    report[model_name] = r2_score(y_test, y_test_pred)
                except Exception as grid_error:
                    # Log error for individual models
                    print(f"Error during GridSearchCV for {model_name}: {grid_error}")
                    report[model_name] = None

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
