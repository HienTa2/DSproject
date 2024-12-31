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
