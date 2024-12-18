import os
import time
import yaml
from src.data_ingestion.finnhub_client import FinnhubDataClient

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../configs/data_sources.yaml')
HISTORICAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/historical')


def download_historical_data():
    # Load configuration
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    symbols_config = config.get('symbols', [])
    client = FinnhubDataClient(config_path=CONFIG_PATH)

    os.makedirs(HISTORICAL_DATA_DIR, exist_ok=True)

    for symbol_info in symbols_config:
        symbol = symbol_info['symbol']
        resolution = symbol_info['resolution']
        days_back = symbol_info['days_back']

        print(f"Downloading data for {symbol} - {days_back} days back at resolution {resolution}")


        # Calculate timestamps
        end_time = int(time.time())
        start_time = end_time - days_back * 24 * 60 * 60

        print(f"Downloading data for {symbol} - {days_back} days back at resolution {resolution}")
        df = client.get_historical_data(symbol, resolution, start_time, end_time)

        # Save the data to CSV
        filename = f"{symbol}_{resolution}_{days_back}days.csv"
        filepath = os.path.join(HISTORICAL_DATA_DIR, filename)
        df.to_csv(filepath)
        print(f"Saved {symbol} data to {filepath}")


if __name__ == "__main__":
    print("Running download_data...")
    download_historical_data()
    print("Done.")
