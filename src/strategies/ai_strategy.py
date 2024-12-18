# project/src/strategies/ai_strategy.py

import os
import numpy as np
import pandas as pd
from decimal import Decimal

from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.data.bar import Bar

# Assume we have a trained LSTM model and a base_model interface
from src.models.lstm_model import LSTMModel
from src.data_ingestion.feature_engineering import generate_features


# You can use Nautilus Trader built-in indicators or implement your own EMA calculation.
# For simplicity, let's manually compute EMAs here, but Nautilus Trader has built-in indicators you can register.
# If using built-in indicators, you'd subscribe and update them similarly to how we handle bars.
def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


class AIStrategyConfig(StrategyConfig):
    instrument_id: InstrumentId
    model_path: str
    seq_length: int = 30
    threshold_buy: float = 0.7
    threshold_sell: float = 0.7
    trade_size: str = "1000"  # e.g., "1000" units
    bar_type: str = "1m"  # Adjust if different bar interval is required
    short_ema_period: int = 50
    long_ema_period: int = 200
    stop_loss_pct: float = 0.01  # 1% stop loss from entry price
    take_profit_pct: float = 0.02  # 2% take profit from entry price


class AIStrategy(Strategy):
    def __init__(self, config: AIStrategyConfig):
        super().__init__(config=config)
        self.instrument_id = config.instrument_id
        self.model_path = config.model_path
        self.seq_length = config.seq_length
        self.threshold_buy = config.threshold_buy
        self.threshold_sell = config.threshold_sell
        self.trade_size = Decimal(config.trade_size)
        self.bar_type = config.bar_type
        self.short_ema_period = config.short_ema_period
        self.long_ema_period = config.long_ema_period
        self.stop_loss_pct = config.stop_loss_pct
        self.take_profit_pct = config.take_profit_pct

        self.bars = []
        self.model = None
        self.position = None  # Track current position if needed

    def on_start(self):
        # Load the trained model
        # input_size should match the number of features from feature engineering steps.
        input_size = 12  # Adjust to your actual number of features
        self.model = LSTMModel(input_size=input_size)
        self.model.load(self.model_path)

        # Subscribe to bar data
        self.subscribe_bars(instrument_id=self.instrument_id, bar_type=self.bar_type)

    def on_stop(self):
        self.close_all_positions(self.instrument_id)
        self.unsubscribe_bars(instrument_id=self.instrument_id, bar_type=self.bar_type)

    def on_bar(self, bar: Bar):
        self.bars.append(bar)

        # We need at least seq_length bars to make a prediction
        if len(self.bars) < self.seq_length:
            return

        # Prepare the features for the model
        df_features = self._prepare_features()
        if df_features is None or len(df_features) < self.seq_length:
            return

        seq_data = df_features.iloc[-self.seq_length:]
        X = seq_data.values[np.newaxis, :, :]  # shape (1, seq_length, num_features)

        # Predict: probability that price will go up
        prob_up = self.model.predict(X)[0]

        # Compute EMA signals
        short_ema_value = ema(df_features['close'], self.short_ema_period).iloc[-1]
        long_ema_value = ema(df_features['close'], self.long_ema_period).iloc[-1]

        # Decision logic combining AI and EMA signals:
        # For example: Buy if prob_up > threshold_buy and short EMA > long EMA
        # Sell if prob_up < (1 - threshold_sell) and short EMA < long EMA
        if prob_up > self.threshold_buy and short_ema_value > long_ema_value:
            # Go long if not already long
            if not self._has_open_position(side="LONG"):
                self._enter_long()
        elif (1.0 - prob_up) > self.threshold_sell and short_ema_value < long_ema_value:
            # Go short if not already short
            if not self._has_open_position(side="SHORT"):
                self._enter_short()

        # Manage existing position with stop-loss and take-profit
        self._manage_risk()

    def _prepare_features(self):
        data = {
            'time': [b.timestamp for b in self.bars],
            'open': [float(b.open) for b in self.bars],
            'high': [float(b.high) for b in self.bars],
            'low': [float(b.low) for b in self.bars],
            'close': [float(b.close) for b in self.bars],
            'volume': [float(b.volume) for b in self.bars],
        }
        df = pd.DataFrame(data).set_index('time')

        df_with_features = generate_features(df)

        # Ensure the features are in the correct order and any scaling is applied
        # If you selected specific feature columns during training, do so here.
        # features_order = [...]
        # df_with_features = df_with_features[features_order]

        df_with_features = df_with_features.dropna()
        return df_with_features

    def _has_open_position(self, side: str):
        # Check if we have an open position on this instrument
        positions = self.cache.positions(self.instrument_id)
        if not positions:
            return False
        # For simplicity, assume a single position per instrument
        pos = positions[0]
        if side == "LONG" and pos.side.is_long:
            return True
        if side == "SHORT" and pos.side.is_short:
            return True
        return False

    def _enter_long(self):
        # Close any short positions before going long (if applicable)
        self.close_all_positions(self.instrument_id)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.trade_size,
        )
        self.submit_order(order)

    def _enter_short(self):
        # Close any long positions before going short (if applicable)
        self.close_all_positions(self.instrument_id)
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.trade_size,
        )
        self.submit_order(order)

    def _manage_risk(self):
        # Check if we have a position and if stop-loss or take-profit conditions are met
        positions = self.cache.positions(self.instrument_id)
        if not positions:
            return
        pos = positions[0]

        current_price = float(self.bars[-1].close)
        entry_price = float(pos.entry_price)

        # Calculate P&L
        pnl = (current_price - entry_price) if pos.side.is_long else (entry_price - current_price)
        pnl_pct = pnl / entry_price

        # If loss exceeds stop-loss_pct, close position
        if pnl_pct < -self.stop_loss_pct:
            self.close_position(pos)

        # If gain exceeds take_profit_pct, close position
        if pnl_pct > self.take_profit_pct:
            self.close_position(pos)

    def on_event(self, event):
        # Handle events if needed (e.g., PositionOpened, PositionClosed)
        pass

    def on_dispose(self):
        pass
