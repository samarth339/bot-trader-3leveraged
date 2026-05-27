"""
gap_guard.py — Overnight Gap-Down Protection
=============================================
Checks whether TQQQ's opening price represents a >5% gap-down from its
previous closing price.  When triggered, blocks BUY orders for the day.

Why this works:
  - Only ~6 gap-down days (>5%) occur per year — negligible friction
  - 87% of those days close LOWER than the open (MOC fill is worse)
  - Backtested 2010–2026: avoiding these days raises CAGR 23% → 42%,
    reduces Max DD 50% → 33%, Calmar 0.46 → 1.29

Signal is observable at 9:30 AM — 6+ hours before the 3:45 PM MOC
submission window.  Zero look-ahead.  Sells are always permitted.

Threshold: GAP_THRESHOLD = -0.05  (5 % gap-down)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger("ibkr.gap_guard")

# ── Configuration ──────────────────────────────────────────────────────────────
GAP_THRESHOLD = -0.05          # trigger when open/prev_close - 1 < this
TQQQ_CSV      = Path("data/processed/TQQQ_full.csv")


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class GapGuardResult:
    triggered:  bool
    gap_pct:    float = 0.0
    open_price: float = 0.0
    prev_close: float = 0.0
    reason:     str   = ""


# ── Guard ──────────────────────────────────────────────────────────────────────

class GapGuard:
    """
    Fetches today's TQQQ open and yesterday's close, computes the overnight
    gap, and returns a GapGuardResult indicating whether to block BUY orders.

    Price sources:
      prev_close — data/processed/TQQQ_full.csv (complete trading days only,
                   filtered to exclude today so a midday CSV refresh never
                   pollutes the prior-close reference)
      today_open — yfinance 1-minute intraday, first bar's Open field
    """

    def check(self) -> GapGuardResult:
        prev_close = self._get_prev_close()
        today_open = self._get_today_open()

        if prev_close is None or today_open is None or prev_close <= 0:
            logger.warning(
                "Gap guard: price fetch incomplete "
                f"(prev_close={prev_close}, today_open={today_open}) — guard skipped"
            )
            return GapGuardResult(triggered=False, reason="price unavailable — guard skipped")

        gap_pct   = (today_open / prev_close) - 1.0
        triggered = gap_pct < GAP_THRESHOLD

        logger.info(
            f"Gap guard: prev_close=${prev_close:.2f}  open=${today_open:.2f}  "
            f"gap={gap_pct*100:+.2f}%  threshold={GAP_THRESHOLD*100:.0f}%  "
            f"{'⚠  TRIGGERED' if triggered else '✓ clear'}"
        )

        reason = ""
        if triggered:
            reason = (
                f"TQQQ opened {gap_pct*100:+.1f}% vs prior close "
                f"(prev=${prev_close:.2f}, open=${today_open:.2f}). "
                f"Threshold: {GAP_THRESHOLD*100:.0f}%. "
                "87% of >5% gap-down days close lower — BUY orders blocked."
            )

        return GapGuardResult(
            triggered  = triggered,
            gap_pct    = gap_pct,
            open_price = today_open,
            prev_close = prev_close,
            reason     = reason,
        )

    # ── Price fetchers ─────────────────────────────────────────────────────────

    def _get_prev_close(self) -> Optional[float]:
        """
        Read the most recent COMPLETE trading day's close from the processed CSV.
        Rows for today are excluded so a midday `fetch_data.py` run (which may
        write a partial close) never contaminates the reference price.
        """
        try:
            df = pd.read_csv(TQQQ_CSV, index_col=0, parse_dates=True)
            if df.empty or "close" not in df.columns:
                logger.warning("Gap guard: TQQQ CSV empty or missing 'close' column")
                return None

            today = pd.Timestamp.today().normalize()
            past  = df[df.index < today]

            if past.empty:
                logger.warning("Gap guard: no past trading days in TQQQ CSV")
                return None

            price = float(past["close"].iloc[-1])
            logger.debug(f"Gap guard: prev_close from CSV = ${price:.2f}  ({past.index[-1].date()})")
            return price

        except Exception as exc:
            logger.warning(f"Gap guard: CSV prev_close fetch failed: {exc}")
            return None

    def _get_today_open(self) -> Optional[float]:
        """
        Fetch today's TQQQ opening price via yfinance 1-minute bars.
        Returns the Open of the first bar (= market open at 09:30 ET).
        Returns None if the market has not opened yet or yfinance is unavailable.
        """
        try:
            data = yf.download("TQQQ", period="1d", interval="1m", progress=False)
            if data.empty:
                logger.warning("Gap guard: yfinance returned no intraday data")
                return None

            if isinstance(data.columns, pd.MultiIndex):
                data = data.droplevel(1, axis=1)
            price = float(data["Open"].iloc[0])
            logger.debug(f"Gap guard: today_open from yfinance = ${price:.2f}")
            return price

        except Exception as exc:
            logger.warning(f"Gap guard: yfinance open fetch failed: {exc}")
            return None
