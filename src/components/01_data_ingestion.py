import os
import sys
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

from exception import CustomException  # Your custom exception module

class DataIngestion:
    """
    Handles reading raw data, splitting into train/test sets, and saving to disk.
    """

    def __init__(self, data_path: str, raw_data_path: str, train_data_path: str, test_data_path: str):
        """
        :param data_path: Path to the raw source data (e.g., CSV file).
        :param raw_data_path: Where to save the full raw dataset after reading.
        :param train_data_path: Where to save the train split.
        :param test_data_path: Where to save the test split.
        """
        self.data_path = data_path
        self.raw_data_path = raw_data_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

    def initiate_data_ingestion(self):
        """
        Reads the CSV, creates train/test splits, and writes them out.
        :return: (train_df, test_df)
        """
        try:
            logging.info("Starting data ingestion step.")
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            df = pd.read_csv(self.data_path)
            logging.info("Successfully loaded raw data.")

            # Save the raw dataset for reference
            df.to_csv(self.raw_data_path, index=False)
            logging.info(f"Raw data saved to {self.raw_data_path}.")

            # Split data
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.train_data_path, index=False)
            test_df.to_csv(self.test_data_path, index=False)
            logging.info("Data ingestion completed successfully.")

            return train_df, test_df

        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise CustomException(e, sys)
