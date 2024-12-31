## End to End Machine Learning



### Key Folders

- **artifacts/**: Stores the trained models and preprocessing objects.
- **data/**: Contains raw and split datasets.
- **notebook/**: Includes Jupyter notebooks for exploratory data analysis (EDA) and model training.
- **src_scripts/**: Contains the core Python scripts for each step of the pipeline.
- **pipeline/**: Main orchestrator script to run the entire workflow.

## Pipeline Workflow

### 1. Data Ingestion
**Script**: `data_ingestion.py`

This module reads the dataset, splits it into training and testing sets, and saves the processed files to the `artifacts/` directory.

### 2. Data Transformation
**Script**: `data_transformation.py`

This module preprocesses the data by:
- Imputing missing values for numerical and categorical features.
- Scaling numerical features.
- Encoding categorical features using OneHotEncoder.

### 3. Model Training
**Script**: `model_trainer.py`

This module:
- Loads the preprocessed data.
- Trains multiple machine learning models (e.g., Random Forest, Gradient Boosting, XGBoost, etc.).
- Evaluates models using R² scores.
- Saves the best model to `artifacts/models/model.pkl`.

### 4. Utility Functions
**Script**: `utils.py`

Utility functions include:
- Saving/loading Python objects.
- Evaluating models via `GridSearchCV` or manual hyperparameter tuning.

## Configuration
**File**: `config.json`

Defines the models to train and their respective hyperparameters.

```json
{
    "models": {
        "RandomForestRegressor": {},
        "GradientBoostingRegressor": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.1, 0.05]
        },
        "CustomXGBRegressor": {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 0.01]
        }
    }
}


How to Run
Prerequisites
Python 3.8 or higher
Install dependencies: pip install -r requirements.txt
Steps
Place your dataset in the data/ folder as stud.csv.
Run the pipeline:
bash
Copy code
python src_scripts/components/pipeline.py
Check the artifacts/ folder for the preprocessed objects and the trained model.
Customization
Adding New Models
Add the new model to model_mapping in model_trainer.py.
Update config.json with the model name and its hyperparameters.
Modifying Preprocessing
Edit the get_data_transformer_object() method in data_transformation.py to customize feature engineering.

Logging
Logs are saved in the logs/ directory to track pipeline progress and debugging information.

Contact
For questions or contributions, feel free to contact the maintainer.

Example Output
Best Model: GradientBoostingRegressor
R² Score: 0.85
