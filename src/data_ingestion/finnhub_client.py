import os
import finnhub
import pandas as pd
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../configs/data_sources.yaml')


class FinnhubDataClient:
    def __init__(self, config_path=CONFIG_PATH):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        api_key = config['finnhub']['api_key']
        self.client = finnhub.Client(api_key=api_key)

    def get_historical_data(self, symbol: str, resolution: str, start_timestamp: int,
                            end_timestamp: int) -> pd.DataFrame:
        """
        Fetches historical candlestick data for a given symbol and time range.

        Parameters:
            symbol: e.g. "AAPL", "BINANCE:BTCUSDT"
            resolution: One of [1, 5, 15, 30, 60, 'D', 'W', 'M']
                        (If using Finnhub's standard intervals)
            start_timestamp: UNIX timestamp
            end_timestamp: UNIX timestamp

        Returns:
            Pandas DataFrame with columns: time, open, high, low, close, volume
        """
        # Fetch candlestick data
        res = self.client.stock_candles(symbol, resolution, start_timestamp, end_timestamp)

        if res['s'] != 'ok':
            raise ValueError(f"Error fetching data from Finnhub: {res}")

        df = pd.DataFrame({
            'time': pd.to_datetime(res['t'], unit='s'),
            'open': res['o'],
            'high': res['h'],
            'low': res['l'],
            'close': res['c'],
            'volume': res['v']
        })
        df.set_index('time', inplace=True)
        return df
