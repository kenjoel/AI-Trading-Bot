import time
from src.data_ingestion.finnhub_client import FinnhubDataClient

def test_finnhub_data_fetch():
    client = FinnhubDataClient()
    # Example: fetch AAPL data for the last 1 day at 1-minute resolution
    end_time = int(time.time())
    start_time = end_time - 60 * 60 * 24  # subtract one day in seconds

    df = client.get_historical_data(symbol="AAPL", resolution='1', start_timestamp=start_time, end_timestamp=end_time)
    print(df.head())
    assert not df.empty, "DataFrame should not be empty"
