import sys
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from exception import CustomException  # Your custom exception module

@dataclass
class DataTransformationConfig:
    """
    Holds configuration for transforming data, such as paths for saving
    the preprocessor and other relevant settings.
    """
    preprocessor_obj_file_path: str

class DataTransformation:
    """
    Constructs and applies a preprocessing pipeline, then persists it for reuse.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer(self) -> ColumnTransformer:
        """
        Builds a ColumnTransformer that handles both numeric and categorical features.
        :return: A scikit-learn ColumnTransformer pipeline.
        """
        try:
            numeric_features = ["feature1", "feature2"]  # Adjust based on your dataset
            categorical_features = ["cat_feature1", "cat_feature2"]

            numeric_transformer = Pipeline([
                ("scaler", StandardScaler())
            ])

            categorical_transformer = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", numeric_transformer, numeric_features),
                ("cat_pipeline", categorical_transformer, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            logging.error(f"Failed to create data transformer: {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Fits a transformer on the training data and applies it to both train and test.
        Persists the preprocessor for later use.
        :return: (X_train_transformed, y_train, X_test_transformed, y_test)
        """
        try:
            logging.info("Starting data transformation step.")
            preprocessor = self.get_data_transformer()

            # Separate input features and target
            X_train = train_df.drop("target", axis=1)
            y_train = train_df["target"]
            X_test = test_df.drop("target", axis=1)
            y_test = test_df["target"]

            # Fit on training data, transform both train and test
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Persist the preprocessor
            joblib.dump(preprocessor, self.config.preprocessor_obj_file_path)
            logging.info(f"Preprocessor saved to {self.config.preprocessor_obj_file_path}.")

            return X_train_transformed, y_train, X_test_transformed, y_test

        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)
