import datetime
from typing import Optional, List, Dict, Tuple
import os

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def stock_download(
    stock_symbols: List[str],
    num_years: int,
    start_date: Optional[datetime.datetime] = None,
    end_date: Optional[datetime.datetime] = None
) -> Dict[str, pd.DataFrame]:
    """
    Download stock data for given symbols and date range.

    Args:
        stock_symbols (List[str]): The stock symbols to download data for.
        num_years (int): The number of years of data to download.
        start_date (Optional[datetime.datetime]): The start date for the data range.
            If not provided, defaults to 'num_years' ago from today.
        end_date (Optional[datetime.datetime]): The end date for the data range.
            If not provided, defaults to the current date and time.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the downloaded stock data for each symbol.

    Raises:
        ValueError: If any stock symbol is invalid or if there's an error downloading the data.
    """
    end_date = end_date or datetime.datetime.now()
    start_date = start_date or end_date - datetime.timedelta(days=365 * num_years)

    try:
        stock_data = yf.download(stock_symbols, start=start_date, end=end_date)
        if stock_data.empty:
            raise ValueError(f"No data found for stock symbols: {stock_symbols}")
        
        # Drop the 'Adj Close' column
        stock_data = stock_data.drop('Adj Close', axis=1, level=0)
        
        result = {}
        for symbol in stock_symbols:
            df = stock_data.xs(symbol, axis=1, level=1)
            print(f"\nData types for {symbol}:")
            print(df.dtypes)
            result[symbol] = df
        
        return result
    except Exception as e:
        raise ValueError(f"Error downloading stock data: {str(e)}") from e


def plot_stocks(data: Dict[str, pd.DataFrame], save_path: Optional[str] = None) -> Figure:
    """
    Plot the closing price of multiple stocks over time.

    Args:
        data (Dict[str, pd.DataFrame]): Dictionary containing stock data with a 'Close' column for each stock.
        save_path (Optional[str]): Path to save the plot. If None, the plot will be displayed.

    Returns:
        Figure: The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for stock_symbol, stock_data in data.items():
        ax.plot(stock_data.index, stock_data['Close'], label=f"{stock_symbol} Price")
    
    ax.set_title("Stock Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    
    _save_or_show_plot(fig, save_path)
    return fig


def calculate_technical_indicators(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate various technical indicators for multiple stocks.

    Args:
        data (Dict[str, pd.DataFrame]): Dictionary containing stock data with 'Open', 'High', 'Low', 'Close', and 'Volume' columns for each stock.

    Returns:
        Dict[str, pd.DataFrame]: Updated dictionary with technical indicators added to each stock's DataFrame.
    """
    def calculate_for_stock(df):
        # Calculate VWAP
        df['VWAP'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()
        
        # Calculate OBV
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        
        # Calculate Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Calculate Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        return df

    return {symbol: calculate_for_stock(df) for symbol, df in data.items()}


def plot_technical_indicators(data: Dict[str, pd.DataFrame], save_path: Optional[str] = None) -> Figure:
    """
    Plot various technical indicators for multiple stocks.

    Args:
        data (Dict[str, pd.DataFrame]): Dictionary containing stock data with technical indicators for each stock.
        save_path (Optional[str]): Path to save the plot. If None, the plot will be displayed instead.

    Returns:
        Figure: The matplotlib Figure object containing the plot.
    """
    num_stocks = len(data)
    fig, axs = plt.subplots(num_stocks, 3, figsize=(40, 8 * num_stocks), squeeze=False)
    
    for idx, (stock_symbol, df) in enumerate(data.items()):
        # Plot 1: Price, SMA50, SMA200, VWAP
        axs[idx, 0].plot(df.index, df['Close'], label='Close Price')
        axs[idx, 0].plot(df.index, df['SMA_50'], label='SMA 50')
        axs[idx, 0].plot(df.index, df['SMA_200'], label='SMA 200')
        axs[idx, 0].plot(df.index, df['VWAP'], label='VWAP')
        axs[idx, 0].set_title(f'{stock_symbol} - Price, Moving Averages, and VWAP')
        axs[idx, 0].legend()
        
        # Plot 2: RSI
        axs[idx, 1].plot(df.index, df['RSI'], label='RSI')
        axs[idx, 1].axhline(y=70, color='r', linestyle='--')
        axs[idx, 1].axhline(y=30, color='g', linestyle='--')
        axs[idx, 1].set_title(f'{stock_symbol} - RSI')
        axs[idx, 1].legend()
        
        # Plot 3: MACD and OBV
        ax1 = axs[idx, 2]
        ax2 = ax1.twinx()
        ax1.plot(df.index, df['MACD'], label='MACD', color='b')
        ax1.plot(df.index, df['Signal_Line'], label='Signal Line', color='r')
        ax2.plot(df.index, df['OBV'], label='OBV', color='g', alpha=0.5)
        ax1.set_title(f'{stock_symbol} - MACD and OBV')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    _save_or_show_plot(fig, save_path)
    return fig


def _save_or_show_plot(fig: Figure, save_path: Optional[str]) -> None:
    """
    Helper function to either save the plot to a file or display it.

    Args:
        fig (Figure): The matplotlib Figure object to save or display.
        save_path (Optional[str]): Path to save the plot. If None, the plot will be displayed.
    """
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    else:
        plt.show()

