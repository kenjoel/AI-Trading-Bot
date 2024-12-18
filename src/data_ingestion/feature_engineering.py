# project/src/data_ingestion/feature_engineering.py

import os
import glob
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

HISTORICAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/historical')
FEATURE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/features')

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns ['open', 'high', 'low', 'close', 'volume'],
    compute some technical indicators and return an enriched DataFrame.
    """
    # Ensure we have the expected columns
    required_cols = {'open', 'high', 'low', 'close', 'volume'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Compute RSI
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['rsi'] = rsi_indicator.rsi()

    # Compute MACD
    macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_signal'] = macd_indicator.macd_signal()
    df['macd_hist'] = macd_indicator.macd_diff()

    # Compute Bollinger Bands
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_low'] = bollinger.bollinger_lband()

    return df

def process_all_data_files():
    os.makedirs(FEATURE_DATA_DIR, exist_ok=True)
    csv_files = glob.glob(os.path.join(HISTORICAL_DATA_DIR, '*.csv'))

    for fpath in csv_files:
        df = pd.read_csv(fpath, parse_dates=True, index_col='time')
        # Generate features
        df_with_features = generate_features(df)
        # Construct output filename
        fname = os.path.basename(fpath)
        base, ext = os.path.splitext(fname)
        out_fname = f"{base}_features.csv"
        out_fpath = os.path.join(FEATURE_DATA_DIR, out_fname)
        df_with_features.to_csv(out_fpath)
        print(f"Processed {fpath} -> {out_fpath}")

if __name__ == "__main__":
    process_all_data_files()
