import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app)  # To allow CORS requests from React frontend

# Enable logging
logging.basicConfig(level=logging.INFO)

# LSTM prediction function
def train_and_predict(state, commodity_name, num_days):
    try:
        # Load data from the Excel file
        df = pd.read_excel('agri_data.xlsx', sheet_name=state)
    
        if commodity_name not in df.columns:
            return np.zeros(num_days)  # If commodity is not cultivated, return zeros for the selected number of days.
    
        # Preprocessing steps
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        prices = df[[commodity_name]].values

        # Scaling the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)

        # Prepare training data
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        X_train, y_train = [], []

        for i in range(60, len(train_data)):
            X_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=32)

        # Predict the next 'num_days' (as selected by the user)
        test_input = scaled_data[-60:]  # Taking the last 60 timesteps from the original data
        predicted_prices = []

        for _ in range(num_days):
            # Reshape to match the LSTM model input (1, 60, 1)
            test_input_reshaped = np.reshape(test_input, (1, test_input.shape[0], 1))

            # Predict the next price
            prediction = model.predict(test_input_reshaped)
            predicted_price = scaler.inverse_transform(prediction).flatten()[0]  # Flatten and get the first element
            predicted_prices.append(predicted_price)

            # Update the input for the next prediction by appending the predicted price and removing the oldest price
            new_price_scaled = scaler.transform([[predicted_price]])  # Transform predicted price
            test_input = np.append(test_input[1:], new_price_scaled, axis=0)  # Update the input with the new predicted price 

        return predicted_prices
    except Exception as e:
        logging.error(f"Error in train_and_predict: {str(e)}")
        raise e

# API route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        state = data['state']
        commodity = data['commodity']
        duration = int(data['duration'])
        
        # Predict prices using the LSTM model
        predicted_prices = train_and_predict(state, commodity, duration)
        
        # Return the results as a JSON response
        return jsonify({
            'status': 'success',
            'predictions': predicted_prices
        })
    except Exception as e:
        logging.error(f"Error in predict route: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
