import os
import pandas as pd


def load_data(data_name):
    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # Get the project root directory
    processed_data_path = os.path.join(project_root, 'data')  # Set the path to the processed data directory
    data_file_path = os.path.join(processed_data_path, data_name)  # Set the path to the data file
    data = pd.read_csv(data_file_path)  # Load the data
    return data
