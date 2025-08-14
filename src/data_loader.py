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
        Pattern to match files (e.g., 'USDT', 'USD', 'BTC').

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and columns as asset symbols.
    """

    # Step 1 — Find all matching parquet files
    files = glob(path.join(data_folder, f"*{symbol_pattern}*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found for pattern: {symbol_pattern}")

    # Step 2 — Merge all files into one DataFrame
    merged_df = pd.concat(
        [
            pd.read_parquet(file)[['close_time', 'close']]
              .assign(close_time=lambda x: pd.to_datetime(x['close_time'], unit='ms'))
              .rename(columns={'close': path.splitext(path.basename(file))[0]})
              .set_index('close_time')
              .sort_index()
            for file in files
        ],
        axis=1,
        join='outer'
    )

    return merged_df
