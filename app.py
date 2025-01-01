from flask import Flask, request, render_template
import logging
from src_scripts.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Setup logging
logging.basicConfig(level=logging.INFO)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Extract and validate user inputs
            try:
                data = CustomData(
                    gender=request.form.get('gender'),
                    race_ethnicity=request.form.get('ethnicity'),
                    parental_level_of_education=request.form.get('parental_level_of_education'),
                    lunch=request.form.get('lunch'),
                    test_preparation_course=request.form.get('test_preparation_course'),
                    reading_score=float(request.form.get('reading_score')),
                    writing_score=float(request.form.get('writing_score'))
                )
            except ValueError:
                return render_template('home.html', error="Invalid numeric input for scores. Please try again.")

            # Convert input to DataFrame
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Input DataFrame:\n{pred_df}")

            # Predict using the pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])

        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return render_template('home.html', error="An unexpected error occurred. Please try again.")


if __name__ == "__main__":
    import os
    app.run(host=os.getenv("HOST", "127.0.0.1"), debug=True, port=int(os.getenv("PORT", 5000)))
