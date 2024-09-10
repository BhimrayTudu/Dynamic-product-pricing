from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_file_path = 'linear_regression_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()

    # Extract features for prediction
    comp_price = float(data['comp_price'])
    ad_spend = float(data['ad_spend'])
    market_demand = float(data['market_demand'])
    eco_indicator = float(data['eco_indicator'])
    seasonality = float(data['seasonality'])

    # Create a DataFrame with the new sample (only one row)
    new_sample = pd.DataFrame({
        'Competitor Price': [comp_price],
        'Advertising Spend': [ad_spend],
        'Market Demand': [market_demand],
        'Economic Indicator': [eco_indicator],
        'Seasonality Factor': [seasonality]
    })

    # Make prediction using the loaded model
    predicted_price = model.predict(new_sample)

    # Return prediction as JSON response
    return jsonify({'predicted_price': predicted_price[0]})

if __name__ == '__main__':
    app.run(debug=True)

