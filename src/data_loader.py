import pandas as pd
from glob import glob
from os import path, makedirs

def load_and_prepare_prices(data_folder="../data/raw_data", symbol_pattern="USDT"):
    """
    Loads and merges multiple parquet files containing price data for the given symbol pattern.

    Parameters
    ----------
    data_folder : str
        Path to the folder containing parquet files.
    symbol_pattern : str
        Pattern to match files (e.g., 'USDT', 'BTC').

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns as asset symbols.
    """
    # Find all parquet files in the data folder matching the symbol pattern
    files = glob(path.join(data_folder, f"*{symbol_pattern}*.parquet"))

    # If no files are found, raise an error to alert the user
    if not files:
        raise FileNotFoundError(f"No parquet files found for pattern: {symbol_pattern}")

    # Read and merge all files into a single DataFrame
    merged_df = pd.concat(
        [
            pd.read_parquet(file)[['close_time', 'close']]
              .assign(close_time=lambda x: pd.to_datetime(x['close_time'], unit='ms').dt.normalize())
              .rename(columns={'close': path.splitext(path.basename(file))[0]})
              .set_index('close_time')
              .sort_index()
            for file in files
        ],
        axis=1,
        join='outer'
    )

    return merged_df

def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily log returns for all assets.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with daily prices, datetime index, and asset symbols as columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with daily log returns.
    """
    returns = prices.pct_change(fill_method=None).dropna(how='all')
    return returns

def group_returns_by_month(returns: pd.DataFrame):
    """
    Groups a daily returns DataFrame by month, keeping only coins
    with complete data for the month (no NaNs).

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns, datetime index, coins as columns

    Returns
    -------
    monthly_groups : dict
        Keys = month as string ('YYYY-MM'), values = DataFrame with only complete coins
    """
    # Create month column
    returns = returns.copy()
    returns['month'] = returns.index.to_period('M')

    # Dictionary to hold monthly groups
    monthly_groups = {}

    for month, group in returns.groupby('month'):
        # Drop the 'month' column and any coin with NaN in this month
        group = group.drop(columns='month').dropna(axis=1, how='any')
        # Save to dictionary
        monthly_groups[str(month)] = group

    return monthly_groups

def save_monthly_returns(monthly_returns: dict, out_folder="../data/processed"):
    """
    Saves monthly return DataFrames into the processed folder, 
    one file per month (Parquet).
    """

    makedirs(out_folder, exist_ok=True)

    for month, df in monthly_returns.items():
        file_path = path.join(out_folder, f"monthly_returns_{month}.parquet")
        df.to_parquet(file_path)
