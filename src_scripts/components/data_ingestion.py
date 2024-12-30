import os
import sys
from src_scripts.exception import CustomException
from src_scripts.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src_scripts.components.data_transformation import DataTransformation
from src_scripts.components.data_transformation import DataTransformationConfig
from src_scripts.components.model_trainer import ModelTrainerConfig
from src_scripts.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'data', "train.csv")
    test_data_path: str = os.path.join('artifacts', 'data', "test.csv")
    raw_data_path: str = os.path.join('artifacts', 'data', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r"C:\Users\Admin\PycharmProjects\DSproject\notebook\data\stud.csv")
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Data Ingestion Step
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Data Transformation Step
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model Training Step
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
