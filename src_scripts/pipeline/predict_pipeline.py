import sys
import pandas as pd
from src_scripts.exception import CustomException
from src_scripts.utils import load_object
import logging
import os


class PredictPipeline:
    def __init__(self):
        # Define paths to preprocessor and model artifacts
        self.preprocessor_path = os.path.join("src_scripts", "components", "artifacts", "transformers", "preprocessor.pkl")
        self.model_path = os.path.join("src_scripts", "components", "artifacts", "models", "model.pkl")

    def predict(self, data_frame):
        try:
            # Validate file paths
            if not os.path.exists(self.preprocessor_path):
                logging.error(f"Preprocessor path not found: {self.preprocessor_path}")
                raise FileNotFoundError(f"Preprocessor path not found: {self.preprocessor_path}")
            if not os.path.exists(self.model_path):
                logging.error(f"Model path not found: {self.model_path}")
                raise FileNotFoundError(f"Model path not found: {self.model_path}")

            # Load the preprocessor and model
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)
            logging.info("Successfully loaded preprocessor and model.")

            # Validate input DataFrame
            required_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch",
                                 "test_preparation_course", "reading_score", "writing_score"]
            missing_columns = [col for col in required_columns if col not in data_frame.columns]
            if missing_columns:
                logging.error(f"Missing columns in input data: {missing_columns}")
                raise ValueError(f"Missing columns: {missing_columns}")

            # Preprocess the data and predict
            logging.info(f"Input DataFrame:\n{data_frame}")
            data = preprocessor.transform(data_frame)
            logging.info(f"Transformed data: {data}")

            predictions = model.predict(data)
            logging.info(f"Predictions: {predictions}")

            return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        # Initialize with input fields
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Create a dictionary from input data
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert dictionary to a DataFrame
            data_frame = pd.DataFrame(data_dict)
            logging.info(f"Generated input DataFrame:\n{data_frame}")
            return data_frame
        except Exception as e:
            logging.error(f"Error creating DataFrame from input data: {e}")
            raise CustomException(e, sys)
