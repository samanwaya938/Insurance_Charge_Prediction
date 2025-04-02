from flask import Flask, render_template, request
from src.pipeline.prediction_pipeline import PredictionPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "GET":
        return render_template("index.html")
    else:
        # Get form data for insurance features
        data = CustomData(
            age=int(request.form.get('age')),
            sex=request.form.get('sex'),
            bmi=float(request.form.get('bmi')),
            children=int(request.form.get('children')),
            smoker=request.form.get('smoker'),
            region=request.form.get('region')
        )

        # Convert the data to a DataFrame
        final_data = data.get_data_as_dataframe()
        print(final_data)

        # Initialize the prediction pipeline
        predict_pipeline = PredictionPipeline()

        # Make the prediction
        prediction = predict_pipeline.predict(final_data)
        print(prediction)

        # Format the prediction for display (assuming prediction is a float)
        prediction_text = f"Predicted Insurance Charge: Rs.{prediction[0]:.2f}"

        # Render the template with the prediction
        return render_template('index.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)