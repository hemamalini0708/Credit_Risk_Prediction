# app.py (Flask app)
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (pkl format)
try:
    with open('Credit_card.pkl', 'rb') as f: # replace 'credit_risk_model.pkl' with your model path
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: Model file 'credit_risk_model.pkl' not found.")
    model = None  # Handle the case where the model file is missing

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}) #handle model not loaded error

    try:
        data = request.get_json()
        input_data = np.array([
            data['DebtRatio'],
            data['NumberOfOpenCreditLinesAndLoans'],
            data['NumberRealEstateLoansOrLines'],
            data['MonthlyIncome_ran_sam'],
            data['NumberOfDependents_ran_sam'],
            data['Education'],
            data['Region_Central'],
            data['Region_East'],
            data['Region_North'],
            data['Region_West']
        ]).reshape(1, -1)

        prediction = model.predict(input_data)[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)