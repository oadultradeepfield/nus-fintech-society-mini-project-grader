import numpy as np
import pandas as pd
import tensorflow as tf
import base64
import json

# Constants for Dataset
MEAN = 0.13175272943327238
STD = 0.09201894076718117

# Paths for Dataset
NON_TEST_DATA_PATH = "data/dogecoin.csv"
TEST_DATA_PATH = "data/dogecoin_test.csv"

# First Trading Date
FIRST_TRADING_DATE = "2021-04-03"

def create_sequences(data: np.array, 
                     sequence_length: int = 7) -> np.array:
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

def calculate_percentage_change(initial_balance: float, 
                                final_balance: float) -> float:
    """
    Calculates the percentage change between an initial and final balance.

    Args:
        initial_balance (float): The starting value of the balance.
        final_balance (float): The ending value of the balance.

    Returns:
        float: The percentage change from the initial balance to the final balance. 
               A positive value indicates an increase, while a negative value indicates a decrease.
    """
    return (final_balance - initial_balance) / initial_balance * 100

    
def generate_output(name: str, 
                    nus_id: str, 
                    initial_balance: float, 
                    final_balance: float) -> str:
    """
    Generates a secret code by encoding input parameters into a Base64 string.

    Args:
        name (str): The name of the user or entity.
        nus_id (str): The NUS ID of the user or entity.
        initial_balance (float): The initial balance represented as a floating-point number.
        final_balance (float): The final balance represented as a floating-point number.

    Returns:
        str: A URL-safe Base64 encoded string representing the secret code.
    """
    data = {
        "name": name.strip(),
        "nus_id": nus_id.strip(),
        "initial_price": initial_balance,
        "final_price": final_balance
    }
    
    json_data = json.dumps(data)
    secret_code = base64.urlsafe_b64encode(json_data.encode()).decode('utf-8')
    return secret_code

def decode_secret_code(secret_code: str) -> dict:
    """
    Decodes a Base64 encoded secret code back into its original components.

    Args:
        secret_code (str): A Base64 encoded string representing the secret code.

    Returns:
        dict: A dictionary containing the original input parameters:
              - 'name': The name of the user or entity.
              - 'nus_id': The NUS ID of the user or entity.
              - 'initial_balance': The initial balance as a string.
              - 'final_balance': The final balance as a float.
    """
    json_data = base64.urlsafe_b64decode(secret_code).decode('utf-8')
    data = json.loads(json_data)
    return data

# Test the script
if __name__ == "__main__":
    trading_data = create_simulate_trading_data()
    model = tf.keras.models.load_model("model/baseline_model.h5")
    
    name = "John Doe"
    nus_id = "e1234567"
    initial_balance = 1500.0
    final_balance = simulate_trading(trading_data, model, initial_balance)
    percentage_change = calculate_percentage_change(initial_balance, final_balance)
    encoded_output = generate_output(name, nus_id, initial_balance, final_balance)
    
    print("Initial Balance:", initial_balance)
    print("Final Balance:", final_balance)
    print("Percentage Change:", percentage_change)
    print("Encoded Output:", encoded_output)
    print("Decoded Output:", decode_secret_code(encoded_output))
