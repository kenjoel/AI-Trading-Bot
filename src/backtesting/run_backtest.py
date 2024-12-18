# project/src/backtesting/run_backtest.py

import os
from decimal import Decimal
import pandas as pd

from nautilus_trader.backtest.node import (
    BacktestDataConfig,
    BacktestEngineConfig,
    BacktestNode,
    BacktestRunConfig,
    BacktestVenueConfig,
)
from nautilus_trader.config import ImportableStrategyConfig
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.core.datetime import dt_to_unix_nanos
from nautilus_trader.model import QuoteTick  # or Bar if using bar data
from nautilus_trader.model import Venue
from nautilus_trader.test_kit.providers import TestInstrumentProvider

# Adjust these based on your setup
CATALOG_PATH = os.path.join(os.getcwd(), "catalog")  # catalog with your data
MODEL_PATH = os.path.join(os.getcwd(), "data/features/lstm_model_v1.pth")
INSTRUMENT_ID = TestInstrumentProvider.btcusdt_binance().id  # Example instrument; adjust as needed

def main():
    # Load instruments from catalog
    catalog = ParquetDataCatalog(CATALOG_PATH)
    instruments = catalog.instruments()
    # Choose the instrument you want to trade
    instrument = [inst for inst in instruments if inst.id == INSTRUMENT_ID][0]

    # Define start/end times for backtest
    # Example: backtest over a specific date range
    start_dt = pd.Timestamp("2023-01-01", tz="UTC")
    end_dt = pd.Timestamp("2023-02-01", tz="UTC")
    start = dt_to_unix_nanos(start_dt)
    end = dt_to_unix_nanos(end_dt)

    # Configure venues (simulated exchange)
    venue_configs = [
        BacktestVenueConfig(
            name="SIM",
            oms_type="NETTING",
            account_type="CASH",
            base_currency="USDT",
            starting_balances=["100000 USDT"]
        ),
    ]

    # Data configuration
    # If using bars, change data_cls to your Bar class and set appropriate parameters.
    data_configs = [
        BacktestDataConfig(
            catalog_path=str(catalog.path),
            data_cls=QuoteTick,           # or Bar if your AI strategy is fed with bars
            instrument_id=instrument.id,
            start_time=start,
            end_time=end,
        ),
    ]

    # Strategy configuration
    # We'll use the AIStrategy from ai_strategy.py with chosen parameters
    strategies = [
        ImportableStrategyConfig(
            strategy_path="src.strategies.ai_strategy:AIStrategy",
            config_path="src.strategies.ai_strategy:AIStrategyConfig",
            config={
                "instrument_id": instrument.id,
                "model_path": MODEL_PATH,
                "seq_length": 30,
                "threshold_buy": 0.7,
                "threshold_sell": 0.7,
                "trade_size": "1000",
                "bar_type": "1m",
                "short_ema_period": 50,
                "long_ema_period": 200,
                "stop_loss_pct": 0.01,
                "take_profit_pct": 0.02
            },
        ),
    ]

    # Engine config
    engine_config = BacktestEngineConfig(strategies=strategies)

    # BacktestRunConfig ties it all together
    config = BacktestRunConfig(
        engine=engine_config,
        data=data_configs,
        venues=venue_configs,
    )

    # Run backtest
    node = BacktestNode(configs=[config])
    results = node.run()

    # Extract the backtest result
    result = results[0]
    engine = node.get_engine(config.id)

    # Generate basic reports
    fills_report = engine.trader.generate_order_fills_report()
    positions_report = engine.trader.generate_positions_report()
    account_report = engine.trader.generate_account_report(Venue("SIM"))

    print("Order Fills Report:")
    print(fills_report)

    print("Positions Report:")
    print(positions_report)

    print("Account Report:")
    print(account_report)

    # Here you could compute additional metrics like Sharpe ratio, max drawdown, directional accuracy
    # Example: directional accuracy calculation if you have predictions and actual moves
    # (This might require saving predictions and comparing them to next period returns.)

if __name__ == "__main__":
    main()
