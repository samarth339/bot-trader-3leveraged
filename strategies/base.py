"""Base class for all strategies."""


class BaseStrategy:
    """
    Every strategy must implement:
        generate_signal(context) → dict
        reset()

    Context dict keys (provided by engine):
        date, index, dates, tqqq, sqqq, qqq,
        position, equity, drawdown, cash
    """

    def generate_signal(self, context: dict) -> dict:
        raise NotImplementedError

    def reset(self):
        """Called before each backtest run to clear any state."""
        pass

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def roc(series, window: int) -> float:
        """Rate of Change over `window` bars (as a decimal)."""
        if len(series) <= window:
            return 0.0
        return (series.iloc[-1] / series.iloc[-window - 1]) - 1

    @staticmethod
    def sma(series, window: int) -> float:
        if len(series) < window:
            return float(series.mean())
        return float(series.iloc[-window:].mean())

    @staticmethod
    def above_ma(series, window: int) -> bool:
        if len(series) < window:
            return False
        return float(series.iloc[-1]) > (series.iloc[-window:].mean())

    @staticmethod
    def rsi(series, window: int = 14) -> float:
        """Simple RSI (non-smoothed) over `window` bars."""
        if len(series) < window + 2:
            return 50.0
        delta = series.diff().dropna().iloc[-(window + 1):]
        gain  = delta.clip(lower=0).mean()
        loss  = (-delta.clip(upper=0)).mean()
        if loss == 0:
            return 100.0
        return float(100 - 100 / (1 + gain / loss))

    @staticmethod
    def atr(df, window: int = 14) -> float:
        """Average True Range from a DataFrame with high/low/close columns."""
        if len(df) < window + 1:
            return float("nan")
        high      = df["high"].iloc[-(window + 1):]
        low       = df["low"].iloc[-(window + 1):]
        close     = df["close"].iloc[-(window + 1):]
        prev_close = close.shift(1)
        import pandas as pd
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return float(tr.iloc[1:].mean())
