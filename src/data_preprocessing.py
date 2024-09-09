import pandas as pd
import numpy as np
from typing import List

def interpolate_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate NaN values in the DataFrame, tailored for stock data.
    
    Args:
    df (pd.DataFrame): Input DataFrame with potential NaN values.
    
    Returns:
    pd.DataFrame: DataFrame with NaN values interpolated.
    """
    df_interpolated: pd.DataFrame = df.copy()
    
    columns_to_interpolate: List[str] = df_interpolated.columns.tolist()
    
    for column in columns_to_interpolate:
        if column == 'OBV':
            # For OBV, use forward fill as it's a cumulative indicator
            df_interpolated[column] = df_interpolated[column].ffill()
        elif column in ['SMA_50', 'SMA_200', 'VWAP']:
            # For SMAs and VWAP, use linear interpolation
            df_interpolated[column] = df_interpolated[column].interpolate(method='linear')
        elif column in ['RSI', '%K', '%D']:
            # For RSI and Stochastic Oscillator, use linear interpolation
            df_interpolated[column] = df_interpolated[column].interpolate(method='linear')
            # Ensure values are within 0-100 range
            df_interpolated[column] = df_interpolated[column].clip(0, 100)
        elif column in ['MACD', 'Signal_Line']:
            # For MACD and Signal Line, use linear interpolation
            df_interpolated[column] = df_interpolated[column].interpolate(method='linear')
        elif column in ['BB_Middle', 'BB_Upper', 'BB_Lower']:
            # For Bollinger Bands, use linear interpolation
            df_interpolated[column] = df_interpolated[column].interpolate(method='linear')
        else:
            # For other columns (Close, High, Low, Open, Volume), use linear interpolation
            df_interpolated[column] = df_interpolated[column].interpolate(method='linear')
    
    # Handle any remaining NaNs at the beginning
    df_interpolated = df_interpolated.bfill()
    
    return df_interpolated


def create_target_metric(df: pd.DataFrame, 
                         price_column: str = 'Close', 
                         periods_ahead: int = 1, 
                         threshold: float = 0.0) -> pd.DataFrame:
    """
    Create a binary target metric based on future price movement.
    
    Args:
    df (pd.DataFrame): Input DataFrame with stock price data, date as index.
    price_column (str): Name of the column containing the price data. Default is 'Close'.
    periods_ahead (int): Number of periods to look ahead for price movement. Default is 1.
    threshold (float): The threshold for considering a movement significant. Default is 0.0.
    
    Returns:
    pd.DataFrame: DataFrame with the new target column added.
    """
    df_with_target: pd.DataFrame = df.copy()
    
    # Ensure the index is datetime type
    df_with_target.index = pd.to_datetime(df_with_target.index)
    
    # Sort the DataFrame by date
    df_with_target.sort_index(inplace=True)
    
    # Calculate future return
    future_return: pd.Series = df_with_target[price_column].pct_change(periods=periods_ahead).shift(-periods_ahead)
    
    # Create binary target (1 for price increase, 0 for decrease or no change)
    df_with_target['target'] = (future_return > threshold).astype(int)
    
    # Remove last rows where target cannot be calculated
    df_with_target = df_with_target.iloc[:-periods_ahead]
    
    return df_with_target


def create_lagged_features(df: pd.DataFrame, columns: List[str], lag_periods: List[int]) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Args:
    df (pd.DataFrame): Input DataFrame with time series data.
    columns (List[str]): List of column names to create lags for.
    lag_periods (List[int]): List of lag periods to create.
    
    Returns:
    pd.DataFrame: DataFrame with new lagged features added.
    """
    df_with_lags: pd.DataFrame = df.copy()
    
    for col in columns:
        for lag in lag_periods:
            df_with_lags[f'{col}_lag_{lag}'] = df_with_lags[col].shift(lag)
    
    # Remove rows with NaN values created by lagging
    df_with_lags.dropna(inplace=True)
    
    return df_with_lags


def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the time series data into training and testing sets.
    
    Args:
    df (pd.DataFrame): Input DataFrame with time series data, date as index.
    test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
    
    Returns:
    tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
    """
    # Ensure the index is datetime type and sorted
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Calculate the split point
    split_index = int(len(df) * (1 - test_size))
    
    # Split the data
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    return train_df, test_df

# Example usage:
# aapl_df_interpolated: pd.DataFrame = interpolate_nans(aapl_df)
# amzn_df_interpolated: pd.DataFrame = interpolate_nans(amzn_df)
# nvda_df_interpolated: pd.DataFrame = interpolate_nans(nvda_df)
# tsla_df_interpolated: pd.DataFrame = interpolate_nans(tsla_df)

# aapl_df_with_target: pd.DataFrame = create_target_metric(aapl_df, periods_ahead=5)
# amzn_df_with_target: pd.DataFrame = create_target_metric(amzn_df, periods_ahead=5)
# nvda_df_with_target: pd.DataFrame = create_target_metric(nvda_df, periods_ahead=5)
# tsla_df_with_target: pd.DataFrame = create_target_metric(tsla_df, periods_ahead=5)

# columns_to_lag = ['Close', 'Volume', 'RSI']
# lag_periods = [1, 5, 10]
# aapl_df_with_lags = create_lagged_features(aapl_df_with_target, columns_to_lag, lag_periods)

# Example usage:
# aapl_train, aapl_test = train_test_split_time_series(aapl_df_with_lags)
# amzn_train, amzn_test = train_test_split_time_series(amzn_df_with_lags)
# nvda_train, nvda_test = train_test_split_time_series(nvda_df_with_lags)
# tsla_train, tsla_test = train_test_split_time_series(tsla_df_with_lags)
