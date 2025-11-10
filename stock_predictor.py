import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# --- Configuration ---
TICKER = "SPY"
START_DATE = "2015-01-01"
END_DATE = "2024-01-01"
LOOKBACK_WINDOW = 60  # Number of past days to look at (sequence length)
FORECAST_HORIZON = 1  # Number of future days to predict
TEST_SIZE = 0.2
EPOCHS = 20
BATCH_SIZE = 32

# --- 1. Data Fetching ---
def fetch_data(ticker, start, end):
    """Fetches historical stock data using yfinance."""
    print(f"Fetching data for {ticker} from {start} to {end}...")
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        raise ValueError(f"Could not fetch data for {ticker}. Check the ticker and date range.")
    # We will use the 'Close' price for prediction
    return data[['Close']]

# --- 2. Data Preprocessing ---
def prepare_data(data, lookback, horizon):
    """Scales data and creates sequences for time series prediction."""
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    
    # Create sequences
    for i in range(lookback, len(scaled_data) - horizon + 1):
        # X is the sequence of 'lookback' days
        X.append(scaled_data[i-lookback:i, 0])
        # y is the price of the day 'horizon' days in the future
        y.append(scaled_data[i + horizon - 1, 0])
        
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for the Transformer/DNN model: (samples, timesteps, features)
    # Since we only have one feature ('Close'), the shape is (samples, lookback, 1)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into training and testing sets
    split_index = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler

# --- 3. Transformer Block (Custom Component) ---
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Creates a single Transformer block."""
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part (Deep Neural Network component)
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# --- 4. Model Definition (Transformer + DNN) ---
def build_model(input_shape):
    """Builds the combined Transformer and Deep Neural Network model."""
    
    # Transformer Hyperparameters
    head_size = 128
    num_heads = 4
    ff_dim = 4
    num_transformer_blocks = 2
    dropout = 0.1
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Add multiple Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)
        
    # Global Average Pooling to flatten the sequence output
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    
    # Deep Neural Network (DNN) layers for final prediction
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    
    # Output layer for a single price prediction
    outputs = Dense(1, activation="linear")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=1e-4),
        metrics=["mae"]
    )
    
    return model

# --- 5. Training and Evaluation ---
def train_and_evaluate(model, X_train, X_test, y_train, y_test, scaler):
    """Trains the model and evaluates its performance."""
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("\n--- Evaluating Model ---")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss (MSE): {loss:.6f}")
    print(f"Test MAE: {mae:.6f}")
    
    # Make predictions
    predicted_scaled = model.predict(X_test)
    
    # Inverse transform to get actual prices
    # We need to reshape the prediction to be 2D for the scaler
    predicted_prices = scaler.inverse_transform(predicted_scaled)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate final error on actual prices
    final_mae = np.mean(np.abs(predicted_prices - actual_prices))
    print(f"Final MAE on actual prices: ${final_mae:.2f}")
    
    # Plotting the results
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, label='Actual Price', color='blue')
    plt.plot(predicted_prices, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'{TICKER} Stock Price Prediction (Transformer + DNN)')
    plt.xlabel('Time (Test Data Index)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('stock_prediction_results.png')
    print("Prediction plot saved to 'stock_prediction_results.png'")
    
    return history

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Fetch Data
        data = fetch_data(TICKER, START_DATE, END_DATE)
        
        # 2. Prepare Data
        X_train, X_test, y_train, y_test, scaler = prepare_data(
            data, LOOKBACK_WINDOW, FORECAST_HORIZON
        )
        
        # 3. Build Model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_model(input_shape)
        model.summary()
        
        # 4. Train and Evaluate
        train_and_evaluate(model, X_train, X_test, y_train, y_test, scaler)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have a stable internet connection and the ticker/date range are valid.")
