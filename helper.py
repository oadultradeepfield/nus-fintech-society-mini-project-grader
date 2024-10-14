import numpy as np
import pandas as pd
import tensorflow as tf

# Constants for Dataset
MEAN = 0.13175272943327238
STD = 0.09201894076718117

# Paths for Dataset
NON_TEST_DATA_PATH = "data/dogecoin.csv"
TEST_DATA_PATH = "data/dogecoin_test.csv"

# First Trading Date
FIRST_TRADING_DATE = "2021-04-03"

def create_sequences(data: np.array, sequence_length: int = 7) -> np.array:
    """
    Creates sequences of data for time series forecasting.

    Args:
        data (np.array): The input data as a NumPy array.
        sequence_length (int): The length of each sequence to be created (default is 7).

    Returns:
        np.array: A 3D NumPy array where each sequence is of shape 
                   (sequence_length, 1). The shape of the output array will be 
                   (number_of_sequences, sequence_length, 1).
    """
    X = []

    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
    
    return np.array(X).reshape((len(X), sequence_length, 1))


def create_simulate_trading_data() -> pd.DataFrame:
    """
    Creates a DataFrame containing simulated trading data.

    This function reads the non-test and test trading data from CSV files, 
    concatenates the last 6 entries of the non-test data with the test data,
    standardizes the 'Close' prices, and generates a target variable indicating 
    whether the price will go up the next day.

    Returns:
        pd.DataFrame: A DataFrame containing the trading data with columns for 
                       standardized close prices and target variables.
    """
    non_test_data = pd.read_csv(NON_TEST_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)
    trading_data = pd.concat([non_test_data.tail(6), test_data], ignore_index=True)
    trading_data['Close_Standardized'] = (trading_data['Close'] - MEAN) / STD
    trading_data['Target'] = (trading_data['Close'].shift(-1) > trading_data['Close']).astype(int)
    return trading_data


def simulate_trading(trading_data: pd.DataFrame,
                     model: tf.keras.Model, 
                     initial_balance: float = 500.00) -> float:
    """
    Simulates trading based on predictions from a trained model.

    This function uses the provided trading data and a trained TensorFlow model 
    to simulate buying and selling stocks. It starts with an initial balance, 
    making trades based on the model's predictions, and returns the final balance.

    Args:
        trading_data (pd.DataFrame): The DataFrame containing trading data with 
                                      standardized prices and targets.
        model (tf.keras.Model): The trained TensorFlow model used for predicting 
                                stock price movements.
        initial_balance (float): The initial balance available for trading 
                                 (default is 500.00).

    Returns:
        float: The final balance after simulating trades based on model predictions.
    """
    num_stocks = int(initial_balance // trading_data['Close'].iloc[0])
    final_balance = initial_balance - num_stocks * trading_data['Close'].iloc[0]

    standardized_prices = trading_data['Close_Standardized'].values
    X = create_sequences(standardized_prices)
    
    predictions = model.predict(X)

    for i in range(1, len(predictions) - 1):
        current_price = trading_data['Close'].iloc[i]
        predicted_move = np.round(predictions[i + 1][0])

        if predicted_move == 1:
            continue
        else: 
            if num_stocks > 0:
                final_balance = num_stocks * current_price
                num_stocks = 0.0

    if num_stocks > 0:
        final_balance = num_stocks * trading_data['Close'].iloc[-1]

    return final_balance

# Test the script
if __name__ == "__main__":
    trading_data = create_simulate_trading_data()
    model = tf.keras.models.load_model("model/baseline_model.h5")
    print(simulate_trading(trading_data, model))
