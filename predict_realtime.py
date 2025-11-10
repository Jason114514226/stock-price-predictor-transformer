import yfinance as yf
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# --- Configuration ---
TICKER = "SPY"
LOOKBACK_WINDOW = 60
MODEL_PATH = "stock_predictor_model.keras"
SCALER_PATH = "stock_predictor_scaler.pkl"

# --- 1. Load Model and Scaler ---
def load_assets():
    """Loads the trained model and the MinMaxScaler object."""
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model has been trained and saved by running stock_predictor.py first.")
        return None, None
        
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None, None
        
    return model, scaler

# --- 2. Fetch Latest Data ---
def fetch_latest_data(ticker, lookback):
    """Fetches the required lookback window of data for prediction."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    # Fetch data for a period slightly longer than the lookback window to ensure we get enough data
    start_date = (datetime.now() - timedelta(days=lookback * 2)).strftime('%Y-%m-%d')
    
    print(f"Fetching latest data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"Could not fetch data for {ticker}. Check the ticker and date range.")
    
    # Select features: Open, High, Low, Close, Volume (must match training features)
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[features].tail(lookback)
    
    if len(data) < lookback:
        raise ValueError(f"Not enough data fetched. Only {len(data)} days available, but {lookback} days are required.")
        
    return data

# --- 3. Preprocess and Predict ---
def predict_next_day(model, scaler, latest_data):
    """Scales the data, makes a prediction, and inverse transforms the result."""
    
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
    
    return predicted_price

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Simulated Real-Time Stock Price Prediction ---")
    
    try:
        # 1. Load Assets
        model, scaler = load_assets()
        if model is None or scaler is None:
            exit()
            
        # 2. Fetch Latest Data
        latest_data = fetch_latest_data(TICKER, LOOKBACK_WINDOW)
        
        # 3. Predict
        predicted_price = predict_next_day(model, scaler, latest_data)
        
        print("\n--- Prediction Result ---")
        print(f"Using the last {LOOKBACK_WINDOW} days of data for {TICKER}:")
        print(f"Predicted Closing Price for the next trading day: **${predicted_price:.2f}**")
        
    except Exception as e:
        print(f"An error occurred during real-time prediction: {e}")
