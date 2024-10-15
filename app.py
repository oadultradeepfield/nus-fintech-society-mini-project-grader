import gradio as gr
import tensorflow as tf
from helper import (create_simulate_trading_data, 
                    simulate_trading, 
                    calculate_percentage_change, 
                    generate_output)
from typing import Tuple

def process_inputs(name: str, nus_id: str, initial_balance: float, model_file: gr.File) -> Tuple[str, str, str]:
    """
    Processes the input data to simulate Dogecoin trading and return results.

    Args:
        name (str): The full name of the user.
        nus_id (str): The NUS ID of the user.
        initial_balance (float): The initial balance for the trading simulation.
        model_file (gr.File): The file containing the pre-trained TensorFlow model (.h5).

    Returns:
        Tuple[str, str, str]: A tuple containing:
            - The final balance after trading (formatted as a string with 2 decimal places).
            - The percentage change in balance (formatted as a percentage string).
            - The generated secret key based on input data.
    """
    model = tf.keras.models.load_model(model_file.name)
    trading_data = create_simulate_trading_data()
    final_balance = simulate_trading(trading_data, model, initial_balance)
    percentage_change = calculate_percentage_change(initial_balance, final_balance)
    secret_key = generate_output(name, nus_id, initial_balance, final_balance)
    return f"{final_balance:.2f}", f"{percentage_change:.2f}%", secret_key

iface = gr.Interface(
    theme=gr.themes.Soft(),
    fn=process_inputs,
    inputs=[
        gr.Textbox(label="Full Name", placeholder="John Doe"),
        gr.Textbox(label="NUS-ID", placeholder="e1234567"),
        gr.Number(label="Initial Balance (USD)", minimum=1),
        gr.File(label="Upload .h5 file (TensorFlow model)")
    ],
    outputs=[
        gr.Textbox(label="Final Balance (USD)"), 
        gr.Textbox(label="Percentage Change"),
        gr.Code(label="Your Secret Key")
    ],
    title="🐕 Dogecoin Trading Simulator",
    description="""
    Welcome to the Dogecoin Trading Simulator! This app is designed to grade the mini project for the AY2024/25 NUS Fintech Society Machine Learning Training.

    📝 Instructions:
    1. Enter your personal details in the fields provided.
    2. Upload your TensorFlow model (.h5 file) for Dogecoin trading simulation.
    3. Click 'Submit' to run the test.
    4. The results will automatically appear in the boxes on the right.

    Note: The output boxes are for display only. You don't need to input anything in them.

    Good luck with your submission! 🍀
    """,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
