import sys
import logging
import joblib
from sklearn.linear_model import LogisticRegression  # as an example
from sklearn.metrics import accuracy_score

from exception import CustomException  # Your custom exception module

class ModelTrainer:
    """
    Trains and evaluates a machine learning model, then persists it.
    """

    def __init__(self, model_save_path: str):
        """
        :param model_save_path: Where to save the trained model artifact.
        """
        self.model_save_path = model_save_path

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        """
        Trains a logistic regression model, evaluates it on test data,
        and saves the model to disk.
        :return: Accuracy on the test set.
        """
        logging.info("Starting model training.")
        try:
            model = LogisticRegression()
            model.fit(X_train, y_train)
            logging.info("Model training complete.")

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            logging.info(f"Model validation accuracy: {accuracy:.4f}")

            # Persist the model
            joblib.dump(model, self.model_save_path)
            logging.info(f"Model saved to {self.model_save_path}.")

            return accuracy

        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(e, sys)
