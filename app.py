import yfinance as yf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from flask import Flask, render_template

# --- Configuration ---
TICKER = "SPY"
LOOKBACK_WINDOW = 60
MODEL_PATH = "stock_predictor_model.keras"
SCALER_PATH = "stock_predictor_scaler.pkl"

app = Flask(__name__)

# --- Prediction Logic (Integrated from predict_realtime.py) ---

def get_prediction():
    """Loads model, fetches data, and returns the next day's predicted price."""
    
    # 1. Load Model and Scaler
    try:
        model = load_model(MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        return f"Error loading model or scaler: {e}. Please ensure the model has been trained and saved."

    # 2. Fetch Latest Data
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        # Fetch data for a period slightly longer than the lookback window to ensure we get enough data
        start_date = (datetime.now() - timedelta(days=LOOKBACK_WINDOW * 2)).strftime('%Y-%m-%d')
        
        data = yf.download(TICKER, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return f"Could not fetch data for {TICKER}."
        
        # Select features: Open, High, Low, Close, Volume (must match training features)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        latest_data = data[features].tail(LOOKBACK_WINDOW)
        
        if len(latest_data) < LOOKBACK_WINDOW:
            return f"Not enough data fetched. Only {len(latest_data)} days available, but {LOOKBACK_WINDOW} days are required."
            
    except Exception as e:
        return f"Error fetching data: {e}"

    # 3. Preprocess and Predict
    try:
        # Scale the latest data
        scaled_data = scaler.transform(latest_data)
        
        # Reshape for the model: (samples, timesteps, features)
        X_pred = np.reshape(scaled_data, (1, scaled_data.shape[0], scaled_data.shape[1]))
        
        # Make prediction
        scaled_prediction = model.predict(X_pred, verbose=0)
        
        # Inverse transform the prediction
        # The prediction is for the 'Close' price (index 3)
        dummy_array = np.zeros((1, scaler.n_features_in_))
        dummy_array[:, 3] = scaled_prediction.flatten()
        
        predicted_price = scaler.inverse_transform(dummy_array)[0, 3]
        
        return f"{predicted_price:.2f}"
        
    except Exception as e:
        return f"Error during prediction: {e}"


@app.route('/')
def index():
    """Main dashboard route."""
    
    prediction_result = get_prediction()
    
    # Check if the result is an error message
    if "Error" in prediction_result:
        return render_template('index.html', 
                               ticker=TICKER, 
                               prediction="N/A", 
                               error=prediction_result,
                               moment=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        return render_template('index.html', 
                               ticker=TICKER, 
                               prediction=prediction_result, 
                               error=None,
                               moment=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    # Flask will be run later in a separate shell command
    pass
