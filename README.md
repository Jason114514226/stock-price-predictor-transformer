# AI Agent for Stock Price Prediction using Transformer and DNN

This project implements a deep learning model for time-series stock price prediction, combining the power of the **Transformer architecture** (specifically for sequence modeling) with traditional **Deep Neural Network (DNN)** layers for final regression.

The script fetches historical stock data, preprocesses it, trains the model, and visualizes the prediction results.

## Key Technologies
*   **Python**
*   **TensorFlow/Keras** (for the deep learning model)
*   **yfinance** (for fetching stock data)
*   **Pandas** and **NumPy** (for data manipulation)
*   **Scikit-learn** (for data scaling)

## Setup and Installation

1.  **Clone the repository:**
    \`\`\`bash
    git clone [REPOSITORY_URL]
    cd stock-price-predictor-transformer
    \`\`\`

2.  **Install dependencies:**
    The project requires the following Python libraries. It is highly recommended to use a virtual environment.
    \`\`\`bash
    pip install pandas numpy scikit-learn tensorflow yfinance matplotlib
    \`\`\`

## Usage

The main script is `stock_predictor.py`. You can run it directly from your terminal:

\`\`\`bash
python stock_predictor.py
\`\`\`

### Configuration
You can modify the following constants within `stock_predictor.py` to change the prediction target and training parameters:

| Constant | Description | Default Value |
| :--- | :--- | :--- |
| `TICKER` | The stock ticker symbol to analyze. | `"SPY"` |
| `START_DATE` | Start date for historical data fetching. | `"2015-01-01"` |
| `END_DATE` | End date for historical data fetching. | `"2024-01-01"` |
| `LOOKBACK_WINDOW` | Number of past days used to predict the next day. | `60` |
| `EPOCHS` | Number of training epochs. | `20` |

## Results

After execution, the script will print the model summary and training progress, and save a plot of the actual vs. predicted prices to `stock_prediction_results.png`.

The current results are based on the default configuration:
*   **Test Loss (MSE):** 0.002081
*   **Final MAE on actual prices:** $11.14
\`\`\`
