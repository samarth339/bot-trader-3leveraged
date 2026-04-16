"""
Core backtesting engine — event-driven, daily bar simulation.

Execution models (execution_model parameter):
  "close"      Legacy: fill at same-bar close + slippage.
  "vwap"       Fill at (O+H+L+C)/4 proxy + slippage.  ← more realistic intraday avg
  "next_open"  Signal generated at bar-i close; fill at bar-(i+1) open + slippage.
               Most conservative — matches real-world overnight order workflow.

Usage:
    from backtester.engine import Backtester
    bt = Backtester(tqqq_df, sqqq_df, qqq_df, initial_capital=5000,
                    execution_model="next_open")
    results = bt.run(strategy)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from config.settings import MAX_DRAWDOWN_LIMIT, DAILY_STOP_LOSS

VALID_EXECUTION_MODELS = {"close", "vwap", "next_open"}


# ─────────────────────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    ticker:      str
    shares:      float
    entry_price: float
    entry_date:  pd.Timestamp


@dataclass
class Trade:
    ticker:       str
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    shares:       float
    pnl:          float
    pnl_pct:      float
    exit_reason:  str
    exec_model:   str = "close"   # execution model used for this trade


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio
# ─────────────────────────────────────────────────────────────────────────────

class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.position: Optional[Position] = None
        self.peak_equity     = initial_capital
        self.equity_curve:   list = []
        self.trades:         list = []

    def equity(self, prices: dict) -> float:
        val = self.cash
        if self.position:
            val += self.position.shares * prices.get(self.position.ticker, 0)
        return val

    def drawdown(self, current_equity: float) -> float:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        return (self.peak_equity - current_equity) / self.peak_equity

    def buy(self, ticker: str, price: float, date: pd.Timestamp,
            pct: float = 1.0, exec_model: str = "close"):
        if self.position:
            return
        amount = self.cash * pct
        shares = amount / price
        self.cash -= amount
        self.position = Position(ticker, shares, price, date)

    def sell(self, price: float, date: pd.Timestamp,
             reason: str = "signal", exec_model: str = "close"):
        if not self.position:
            return
        proceeds = self.position.shares * price
        pnl      = proceeds - (self.position.shares * self.position.entry_price)
        pnl_pct  = (price - self.position.entry_price) / self.position.entry_price
        self.trades.append(Trade(
            ticker=self.position.ticker,
            entry_date=self.position.entry_date,
            exit_date=date,
            entry_price=self.position.entry_price,
            exit_price=price,
            shares=self.position.shares,
            pnl=pnl, pnl_pct=pnl_pct,
            exit_reason=reason, exec_model=exec_model,
        ))
        self.cash    += proceeds
        self.position = None

    def sell_partial(self, sell_pct: float, price: float, date: pd.Timestamp,
                     reason: str = "signal_partial", exec_model: str = "close"):
        if not self.position:
            return
        if sell_pct >= 1.0:
            self.sell(price, date, reason, exec_model)
            return
        shares_sold = self.position.shares * sell_pct
        proceeds    = shares_sold * price
        pnl         = proceeds - (shares_sold * self.position.entry_price)
        pnl_pct     = (price - self.position.entry_price) / self.position.entry_price
        self.trades.append(Trade(
            ticker=self.position.ticker,
            entry_date=self.position.entry_date,
            exit_date=date,
            entry_price=self.position.entry_price,
            exit_price=price,
            shares=shares_sold,
            pnl=pnl, pnl_pct=pnl_pct,
            exit_reason=reason, exec_model=exec_model,
        ))
        self.cash            += proceeds
        self.position.shares -= shares_sold
        if self.position.shares * price < 1.0:
            self.position = None


# ─────────────────────────────────────────────────────────────────────────────
#  Backtester
# ─────────────────────────────────────────────────────────────────────────────

class Backtester:
    """
    Day-by-day backtester.

    signal dict from strategy.generate_signal():
        {"action": "buy_tqqq" | "buy_sqqq" | "sell" | "sell_partial" | "hold",
         "size_pct": float,   # fraction of portfolio (default 1.0)
         "sell_pct": float}   # fraction to sell for sell_partial (default 0.5)
    """

    def __init__(
        self,
        tqqq: pd.DataFrame,
        sqqq: pd.DataFrame,
        qqq:  pd.DataFrame,
        initial_capital:      float = 5_000,
        commission_per_share: float = 0.0,
        slippage_pct:         float = 0.001,      # 10 bps
        execution_model:      str   = "close",     # close | vwap | next_open
        # NOTE: "next_open" is most realistic but breaks crash-brake strategies
        # (sell fires, executes next day, condition may re-trigger before re-entry guard).
        # Use "vwap" for a reasonable intraday-average approximation without the lag issue.
        vix: pd.DataFrame = None,
        spy: pd.DataFrame = None,
    ):
        if execution_model not in VALID_EXECUTION_MODELS:
            raise ValueError(
                f"execution_model must be one of {VALID_EXECUTION_MODELS}, got '{execution_model}'"
            )

        # ── Missing data guard ────────────────────────────────────────────────
        for name, df in [("tqqq", tqqq), ("sqqq", sqqq), ("qqq", qqq)]:
            if df is None:
                raise ValueError(f"Required DataFrame '{name}' is None")
            missing = df[["open", "high", "low", "close"]].isnull().sum().sum()
            if missing > 0:
                raise ValueError(
                    f"Missing data detected in '{name}': {missing} null values. "
                    f"Run fetch_data.py to refresh."
                )
        if vix is not None:
            missing_vix = vix["close"].isnull().sum()
            if missing_vix > 0:
                raise ValueError(
                    f"Missing VIX data: {missing_vix} null values — wrong regime possible. "
                    f"Re-run fetch_data.py."
                )

        self.tqqq            = tqqq
        self.sqqq            = sqqq
        self.qqq             = qqq
        self.vix             = vix
        self.spy             = spy
        self.initial_capital = initial_capital
        self.commission      = commission_per_share
        self.slippage        = slippage_pct
        self.execution_model = execution_model

        self.dates = (tqqq.index.intersection(sqqq.index)
                                .intersection(qqq.index)
                                .sort_values())

    # ── Fill price ─────────────────────────────────────────────────────────────
    def _fill_price(self, ticker: str, date: pd.Timestamp,
                    direction: int, model: str = None) -> float:
        """
        Return execution price with slippage.
        direction: +1 = buy (pays up), -1 = sell (receives less)
        model override: if provided, uses that model instead of self.execution_model
        """
        m   = model or self.execution_model
        df  = {"TQQQ": self.tqqq, "SQQQ": self.sqqq, "QQQ": self.qqq}[ticker]
        row = df.loc[date]

        if m == "vwap":
            price = (float(row["open"]) + float(row["high"]) +
                     float(row["low"])  + float(row["close"])) / 4.0
        else:
            # "close" and "next_open" both use close for stops/end-of-backtest;
            # next_open open price is injected directly from the run loop
            price = float(row["close"])

        return price * (1.0 + direction * self.slippage)

    def _open_fill(self, ticker: str, date: pd.Timestamp, direction: int) -> float:
        """Fill at next bar's open with slippage (for next_open execution model)."""
        df    = {"TQQQ": self.tqqq, "SQQQ": self.sqqq, "QQQ": self.qqq}[ticker]
        price = float(df.loc[date, "open"])
        return price * (1.0 + direction * self.slippage)

    # ── Main run loop ──────────────────────────────────────────────────────────
    def run(self, strategy) -> dict:
        portfolio = Portfolio(self.initial_capital)
        strategy.reset()
        em        = self.execution_model

        pending_signal = None   # buffered for next_open model

        for i, date in enumerate(self.dates):

            # ── STEP 1: Execute previous bar's buffered signal (next_open model) ─
            if em == "next_open" and pending_signal is not None:
                p_action   = pending_signal.get("action", "hold")
                p_size_pct = pending_signal.get("size_pct", 1.0)
                p_sell_pct = pending_signal.get("sell_pct", 0.5)

                if p_action == "sell" and portfolio.position:
                    fill = self._open_fill(portfolio.position.ticker, date, -1)
                    portfolio.sell(fill, date, reason="signal", exec_model="next_open")

                elif p_action == "sell_partial" and portfolio.position:
                    fill = self._open_fill(portfolio.position.ticker, date, -1)
                    portfolio.sell_partial(p_sell_pct, fill, date,
                                           reason="signal_partial", exec_model="next_open")

                elif p_action == "buy_tqqq" and not portfolio.position:
                    fill = self._open_fill("TQQQ", date, +1)
                    portfolio.buy("TQQQ", fill, date, pct=p_size_pct, exec_model="next_open")

                elif p_action == "buy_sqqq" and not portfolio.position:
                    fill = self._open_fill("SQQQ", date, +1)
                    portfolio.buy("SQQQ", fill, date, pct=p_size_pct, exec_model="next_open")

            # ── STEP 2: Mark-to-market at close ──────────────────────────────────
            prices = {
                "TQQQ": float(self.tqqq.loc[date, "close"]),
                "SQQQ": float(self.sqqq.loc[date, "close"]),
                "QQQ":  float(self.qqq.loc[date, "close"]),
            }
            equity = portfolio.equity(prices)
            dd     = portfolio.drawdown(equity)

            # ── STEP 3: Hard stops (always same-bar close for speed) ──────────────
            if portfolio.position:
                pos_price = prices[portfolio.position.ticker]
                pos_ret   = ((pos_price - portfolio.position.entry_price)
                             / portfolio.position.entry_price)
                if pos_ret <= -DAILY_STOP_LOSS:
                    fill = self._fill_price(portfolio.position.ticker, date, -1, model="close")
                    portfolio.sell(fill, date, reason="daily_stop_loss", exec_model="stop")
                    pending_signal = None
                elif dd >= MAX_DRAWDOWN_LIMIT:
                    fill = self._fill_price(portfolio.position.ticker, date, -1, model="close")
                    portfolio.sell(fill, date, reason="max_drawdown_circuit_breaker", exec_model="stop")
                    pending_signal = None

            # ── STEP 4: Strategy signal ───────────────────────────────────────────
            vix_value = None
            if self.vix is not None and date in self.vix.index:
                vix_value = float(self.vix.loc[date, "close"])

            context = {
                "date":      date,
                "index":     i,
                "dates":     self.dates,
                "tqqq":      self.tqqq,
                "sqqq":      self.sqqq,
                "qqq":       self.qqq,
                "spy":       self.spy if self.spy is not None else {},
                "vix_value": vix_value,
                "vix_close": self.vix["close"] if self.vix is not None else None,
                "position":  portfolio.position,
                "equity":    equity,
                "drawdown":  dd,
                "cash":      portfolio.cash,
            }
            signal = strategy.generate_signal(context)
            action = signal.get("action", "hold")

            # ── STEP 5: Execute signal ────────────────────────────────────────────
            if em == "next_open":
                # Buffer signal — will execute at tomorrow's open
                pending_signal = signal
            else:
                # Immediate execution at close or VWAP
                size_pct = signal.get("size_pct", 1.0)
                sell_pct = signal.get("sell_pct", 0.5)

                if action == "sell" and portfolio.position:
                    fill = self._fill_price(portfolio.position.ticker, date, -1)
                    portfolio.sell(fill, date, reason="signal", exec_model=em)

                elif action == "sell_partial" and portfolio.position:
                    fill = self._fill_price(portfolio.position.ticker, date, -1)
                    portfolio.sell_partial(sell_pct, fill, date,
                                           reason="signal_partial", exec_model=em)

                elif action == "buy_tqqq" and not portfolio.position:
                    fill = self._fill_price("TQQQ", date, +1)
                    portfolio.buy("TQQQ", fill, date, pct=size_pct, exec_model=em)

                elif action == "buy_sqqq" and not portfolio.position:
                    fill = self._fill_price("SQQQ", date, +1)
                    portfolio.buy("SQQQ", fill, date, pct=size_pct, exec_model=em)

            portfolio.equity_curve.append({"date": date, "equity": equity})

        # ── Close any open position at end ────────────────────────────────────
        if portfolio.position:
            last_date = self.dates[-1]
            fill = self._fill_price(portfolio.position.ticker, last_date, -1, model="close")
            portfolio.sell(fill, last_date, reason="end_of_backtest")

        return self._compile_results(portfolio)

    # ── Results compiler ──────────────────────────────────────────────────────
    def _compile_results(self, portfolio: Portfolio) -> dict:
        ec = pd.DataFrame(portfolio.equity_curve).set_index("date")
        ec["returns"]  = ec["equity"].pct_change()
        ec["peak"]     = ec["equity"].cummax()
        ec["drawdown"] = (ec["peak"] - ec["equity"]) / ec["peak"]

        trades_df = (pd.DataFrame([t.__dict__ for t in portfolio.trades])
                     if portfolio.trades else pd.DataFrame())

        total_return = ec["equity"].iloc[-1] / self.initial_capital - 1
        n_years      = (self.dates[-1] - self.dates[0]).days / 365.25
        cagr         = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        max_dd       = ec["drawdown"].max()
        sharpe       = (ec["returns"].mean() / ec["returns"].std() * np.sqrt(252)
                        if ec["returns"].std() > 0 else 0)
        calmar       = cagr / max_dd if max_dd > 0 else 0

        win_rate = avg_win = avg_loss = profit_factor = 0.0
        if not trades_df.empty:
            wins   = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] <= 0]
            win_rate      = len(wins) / len(trades_df)
            avg_win       = wins["pnl_pct"].mean()   if len(wins)   else 0.0
            avg_loss      = losses["pnl_pct"].mean() if len(losses) else 0.0
            gross_profit  = wins["pnl"].sum()
            gross_loss    = abs(losses["pnl"].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "equity_curve": ec,
            "trades":       trades_df,
            "metrics": {
                "total_return":  total_return,
                "cagr":          cagr,
                "max_drawdown":  max_dd,
                "sharpe":        sharpe,
                "calmar":        calmar,
                "win_rate":      win_rate,
                "avg_win_pct":   avg_win,
                "avg_loss_pct":  avg_loss,
                "profit_factor": profit_factor,
                "n_trades":      len(trades_df),
                "years":         n_years,
                "final_equity":  ec["equity"].iloc[-1],
                "exec_model":    self.execution_model,
            },
        }
