import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, List, Union

def handle_nan_values(data: Dict[str, pd.DataFrame], method: str = 'ffill') -> Dict[str, pd.DataFrame]:
    """
    Handle NaN values in the dataset.
    
    Args:
    data (Dict[str, pd.DataFrame]): Dictionary containing DataFrames for each stock symbol.
    method (str): Method to handle NaN values ('ffill', 'bfill', or 'drop').
    
    Returns:
    Dict[str, pd.DataFrame]: Dictionary containing DataFrames with NaN values handled.
    """
    handled_data: Dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        if method == 'ffill':
            handled_data[symbol] = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'bfill':
            handled_data[symbol] = df.fillna(method='bfill').fillna(method='ffill')
        elif method == 'drop':
            handled_data[symbol] = df.dropna()
        else:
            raise ValueError("Invalid method. Choose 'ffill', 'bfill', or 'drop'.")
    return handled_data

def split_and_scale_data(stock_data: Dict[str, pd.DataFrame], train_ratio: float = 0.75) -> Tuple[Dict[str, Dict[str, Union[np.ndarray, pd.DatetimeIndex]]], Dict[str, MinMaxScaler]]:
    """
    Split the data into training and testing sets, and scale the data using MinMaxScaler.
    
    Args:
    stock_data (Dict[str, pd.DataFrame]): Dictionary containing DataFrames for each stock symbol.
    train_ratio (float): Ratio of data to use for training (default: 0.75).
    
    Returns:
    Tuple[Dict[str, Dict[str, Union[np.ndarray, pd.DatetimeIndex]]], Dict[str, MinMaxScaler]]:
        preprocessed_data (Dict[str, Dict[str, Union[np.ndarray, pd.DatetimeIndex]]]): Dictionary containing scaled train and test data for each symbol.
        scalers (Dict[str, MinMaxScaler]): Dictionary containing fitted MinMaxScaler objects for each symbol.
    """
    preprocessed_data: Dict[str, Dict[str, Union[np.ndarray, pd.DatetimeIndex]]] = {}
    scalers: Dict[str, MinMaxScaler] = {}

    for symbol, data in stock_data.items():
        # Split the data into training and testing sets
        split_index = int(len(data) * train_ratio)
        train_data = data[:split_index]
        test_data = data[split_index:]

        # Initialize and fit MinMaxScaler on training data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data[['Close']])

        # Transform both training and testing data
        train_scaled = scaler.transform(train_data[['Close']])
        test_scaled = scaler.transform(test_data[['Close']])

        preprocessed_data[symbol] = {
            'train': train_scaled,
            'test': test_scaled,
            'train_dates': train_data.index,
            'test_dates': test_data.index
        }
        scalers[symbol] = scaler

    return preprocessed_data, scalers

def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and target values for time series forecasting.
    
    Args:
    data (np.ndarray): Array of scaled stock prices.
    sequence_length (int): Number of time steps to use for each input sequence.
    
    Returns:
    Tuple[np.ndarray, np.ndarray]:
        X (np.ndarray): Input sequences.
        y (np.ndarray): Target values.
    """
    X: List[np.ndarray] = []
    y: List[float] = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

def prepare_data_for_training(preprocessed_data: Dict[str, Dict[str, Union[np.ndarray, pd.DatetimeIndex]]], sequence_length: int = 100) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepare data for model training by creating sequences.
    
    Args:
    preprocessed_data (Dict[str, Dict[str, Union[np.ndarray, pd.DatetimeIndex]]]): Dictionary containing scaled train and test data for each symbol.
    sequence_length (int): Number of time steps to use for each input sequence (default: 100).
    
    Returns:
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        Each element is a dictionary containing data for each stock symbol.
    """
    X_train: Dict[str, np.ndarray] = {}
    y_train: Dict[str, np.ndarray] = {}
    X_test: Dict[str, np.ndarray] = {}
    y_test: Dict[str, np.ndarray] = {}

    for symbol, data in preprocessed_data.items():
        X_train[symbol], y_train[symbol] = create_sequences(data['train'], sequence_length)
        X_test[symbol], y_test[symbol] = create_sequences(data['test'], sequence_length)

    return X_train, y_train, X_test, y_test