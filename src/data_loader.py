import pandas as pd
from glob import glob
from os import path

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

    files = glob(path.join(data_folder, f"*{symbol_pattern}*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found for pattern: {symbol_pattern}")

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

def group_prices_by_month(df: pd.DataFrame) -> dict:
    """
    Groups daily price data into monthly periods, keeping only coins that have data in each month.

    Parameters
    ----------
    df : pd.DataFrame
        Daily prices with datetime index and coin symbols as columns.

    Returns
    -------
    dict
        Dictionary with period keys (YYYY-MM) and monthly DataFrames as values.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Add month period
    df['month'] = df.index.to_period('M')

    # Group by month and drop coins that are not available in that month
    monthly_groups = {}
    for month, group in df.groupby('month'):
        group = group.drop(columns='month')          # remove the 'month' helper column
        group = group.dropna(axis=1, how='all')      # drop coins not available in this month
        monthly_groups[str(month)] = group

    return monthly_groups
