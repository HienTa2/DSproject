from src_scripts.components.data_ingestion import DataIngestion
from src_scripts.components.data_transformation import DataTransformation
from src_scripts.components.model_trainer import ModelTrainer


def main():
    # Step 1: Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Data Transformation
    transformation = DataTransformation()
    train_array, test_array, preprocessor_path = transformation.initiate_data_transformation(train_path, test_path)

    # Step 3: Model Training
    trainer = ModelTrainer()
    results = trainer.initiate_model_trainer(train_array, test_array)

    print("Pipeline Completed!")
    print(f"Best Model: {results['Best Model']}")
    print(f"R² Score: {results['R² Score']}")
    print(results['Explanation'])


if __name__ == "__main__":
    main()