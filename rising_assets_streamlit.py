"""
Rising Assets Strategy — Streamlit Backtester (v6.2a)

Changes in v6.2:
- Adds chart data download for debugging

Changes in v6.2:
- Adds DQ fix for GBp/GBP unit-mix extremes (e.g., SGLN.L prints ~100x too low); logs corrections to Streamlit + Excel Notes.
- Adds "Use max dates" checkbox to automatically determine the maximum common date range
  across all tickers in the universe and benchmark.
- Metrics table now shows both Rising Assets strategy and Benchmark metrics

Fixes vs v5
- Adds guardrail + valuation forward-fill to prevent spurious near-zero equity caused by missing prices.
- Makes start/end inputs robust (explicit min/max + widget keys).
- Makes Streamlit caching keys robust by passing start/end as ISO strings.

Core model
- Monthly rebalance with look-ahead fix:
  * Signal computed as-of prior month-end trading day (t-1)
  * Trades executed at month-end trading day close (t)
- Uses adjusted closes from Yahoo Finance via yfinance (auto_adjust=True).
- Cash earns 0%.

Dependencies
pip install streamlit yfinance pandas numpy xlsxwriter plotly

Optional for embedding PNG charts into Excel:
pip install kaleido
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================
# Utilities
# =========================

def parse_universe(text: str) -> List[str]:
    if not text:
        return []
    s = text.replace("'", "").replace('"', "")
    parts: List[str] = []
    for chunk in s.replace("\n", ",").split(","):
        t = chunk.strip()
        if t:
            parts.append(t)

    out: List[str] = []
    seen = set()
    for t in parts:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def month_end_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading day for each (year, month) present in index."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) == 0:
        return idx
    s = pd.Series(idx, index=idx)
    month_last = s.groupby(idx.to_period("M")).max().sort_values()
    return pd.DatetimeIndex(month_last.values)


# =========================
# Max date range determination
# =========================

def find_max_common_start_date(symbols: List[str]) -> Tuple[date | None, str | None]:
    """
    Fetch maximum available history for all symbols and find the latest
    first-valid-date (so all symbols have data from that point forward).
    Returns (start_date, limiting_symbol)
    """
    if not symbols:
        return None, None

    early_start = date(1990, 1, 1)
    today_date = date.today()
    end_plus = today_date + timedelta(days=1)

    try:
        data = yf.download(
            symbols,
            start=early_start,
            end=end_plus,
            auto_adjust=True,
            progress=False,
            group_by="ticker" if len(symbols) > 1 else None,
        )
        if data is None or len(data) == 0:
            return None, None

        prices = pd.DataFrame()
        if len(symbols) == 1:
            if "Close" not in data.columns:
                return None, None
            prices[symbols[0]] = data["Close"].copy()
        else:
            for t in symbols:
                if hasattr(data.columns, "levels") and t in data.columns.levels[0]:
                    if "Close" in data[t].columns:
                        prices[t] = data[t]["Close"].copy()

        if prices.empty:
            return None, None

        first_dates = {}
        for sym in symbols:
            if sym in prices.columns:
                first_valid = prices[sym].first_valid_index()
                if first_valid is not None:
                    first_dates[sym] = pd.to_datetime(first_valid).date()

        if not first_dates:
            return None, None

        limiting_symbol = max(first_dates, key=first_dates.get)
        common_start = first_dates[limiting_symbol]
        return common_start, limiting_symbol

    except Exception:
        return None, None


# =========================
# Data fetch + cleaning
# =========================

@st.cache_data(show_spinner=False)
def get_yahoo_currency(ticker: str) -> str:
    """Best-effort currency lookup (Yahoo Finance metadata)."""
    try:
        tobj = yf.Ticker(ticker)
    except Exception:
        return ""

    # Try slower but often-complete path
    try:
        info = getattr(tobj, "info", None)
        if isinstance(info, dict):
            ccy = (info.get("currency", "") or "").strip()
            if ccy:
                return ccy
    except Exception:
        pass

    # Try fast path
    try:
        fi = getattr(tobj, "fast_info", None)
        if fi is not None:
            ccy = (getattr(fi, "currency", "") or "").strip()
            if ccy:
                return ccy
    except Exception:
        pass

    return ""


def normalize_prices_to_gbp(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Normalize prices to GBP: if Yahoo reports currency 'GBp' (pence), divide by 100."""
    normalized = prices.copy()
    for ticker in tickers:
        if ticker not in normalized.columns:
            continue
        ccy = get_yahoo_currency(ticker)
        if ccy == "GBp":
            normalized[ticker] = normalized[ticker] / 100.0
    return normalized


def fix_gbp_unit_mix_extremes(
    prices_gbp: pd.DataFrame,
    tickers: List[str],
    factor: float = 100.0,
    ratio_min: float = 50.0,
    ratio_max: float = 150.0,
    tol: float = 0.25,
    rolling_window: int = 30,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Fix ~100x-too-low prints caused by GBp/GBP unit-mix issues.

    This runs AFTER `normalize_prices_to_gbp` (so series are in GBP).

    Failure mode: for a ticker whose Yahoo currency is GBp, Yahoo sometimes returns
    daily prices already in GBP (pounds). Our normalization divides by 100, making
    those days ~100x too low (e.g., 70.55 -> 0.7055). We detect these extremes and
    multiply the bad points by 100.

    Returns (fixed_prices, report_rows).
    """
    fixed = prices_gbp.copy()
    report: List[dict] = []

    for t in tickers:
        if t not in fixed.columns:
            continue

        if get_yahoo_currency(t) != "GBp":
            continue

        s = fixed[t].astype(float)
        v = s[(s.notna()) & (s > 0)]
        if len(v) < 10:
            continue

        roll_med = s.rolling(window=int(rolling_window), min_periods=max(5, int(rolling_window // 2))).median()
        global_med = float(v.median())
        prev = s.shift(1)
        nxt = s.shift(-1)

        for dt, cur in s.items():
            if not (pd.notna(cur) and cur > 0):
                continue

            # Method A: rolling median typical level
            typ = roll_med.loc[dt]
            if not (pd.notna(typ) and typ > 0):
                typ = global_med

            if typ and typ > 0:
                ratio = float(typ / cur)
                if ratio_min <= ratio <= ratio_max and abs(((cur * factor) / typ) - 1.0) <= tol:
                    old_px = float(cur)
                    new_px = float(cur * factor)
                    fixed.at[dt, t] = new_px
                    report.append(
                        {
                            "Ticker": t,
                            "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                            "Currency": "GBp",
                            "Issue": "GBp/GBP unit-mix extreme (corrected)",
                            "Old Price": old_px,
                            "New Price": new_px,
                            "Factor": factor,
                            "Method": "RollingMedian",
                        }
                    )
                    continue

            # Method B: neighbor consistency (best for single-day flips)
            p = prev.loc[dt]
            n = nxt.loc[dt]
            if pd.notna(p) and pd.notna(n) and (p > 0) and (n > 0):
                r1 = float(p / cur)
                r2 = float(n / cur)
                if (ratio_min <= r1 <= ratio_max) and (ratio_min <= r2 <= ratio_max):
                    if (abs(((cur * factor) / p) - 1.0) <= tol) and (abs(((cur * factor) / n) - 1.0) <= tol):
                        old_px = float(cur)
                        new_px = float(cur * factor)
                        fixed.at[dt, t] = new_px
                        report.append(
                            {
                                "Ticker": t,
                                "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                                "Currency": "GBp",
                                "Issue": "GBp/GBP unit-mix extreme (corrected)",
                                "Old Price": old_px,
                                "New Price": new_px,
                                "Factor": factor,
                                "Method": "Neighbors",
                            }
                        )

    return fixed, report


@st.cache_data(show_spinner=False)
def fetch_price_data_robust_cached(tickers: Tuple[str, ...], start_iso: str, end_iso: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices (Close) using yfinance, with fallback downloads.
    Cached in-memory per Streamlit process.
    Uses ISO date strings for stable Streamlit caching keys.
    """
    tlist = [t for t in tickers if t]
    if not tlist:
        raise ValueError("No tickers provided.")

    start_date = pd.Timestamp(start_iso)
    end_date = pd.Timestamp(end_iso)

    try:
        data = yf.download(
            tlist,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            group_by="ticker" if len(tlist) > 1 else None,
            threads=True,
        )
        if data is None or len(data) == 0:
            raise ValueError("No data returned from Yahoo Finance.")

        prices = pd.DataFrame()
        if len(tlist) == 1:
            if "Close" not in data.columns:
                raise ValueError(f"No Close column for {tlist[0]}")
            prices[tlist[0]] = data["Close"].copy()
        else:
            for t in tlist:
                if hasattr(data.columns, "levels") and t in data.columns.levels[0]:
                    if "Close" in data[t].columns:
                        prices[t] = data[t]["Close"].copy()

        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.dropna(axis=1, how="all")
        if prices.shape[1] == 0:
            raise ValueError("Could not extract price data from any ticker.")

        prices = normalize_prices_to_gbp(prices, list(prices.columns))
        return prices

    except Exception:
        prices_dict: Dict[str, pd.Series] = {}
        for t in tlist:
            d = yf.download(
                t,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if d is not None and len(d) > 0 and "Close" in d.columns:
                prices_dict[t] = d["Close"].copy()

        if not prices_dict:
            raise ValueError("No tickers successfully downloaded.")

        prices = pd.DataFrame(prices_dict)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.dropna(axis=1, how="all")
        prices = normalize_prices_to_gbp(prices, list(prices.columns))
        return prices


def validate_and_clean_prices(prices: pd.DataFrame, threshold: float = 0.20) -> Tuple[pd.DataFrame, List[dict]]:
    """Detect and fix single-day spikes."""
    cleaned = prices.copy()
    report: List[dict] = []
    for t in cleaned.columns:
        s = cleaned[t].copy()
        if s.isna().all():
            continue
        pct = s.pct_change()
        pct_next = s.pct_change(-1)
        spikes = (pct.abs() > threshold) & (pct_next.abs() > threshold)
        for dt in s[spikes].index:
            idx = s.index.get_loc(dt)
            if idx <= 0 or idx >= len(s) - 1:
                continue
            prev_px = s.iloc[idx - 1]
            spike_px = s.iloc[idx]
            next_px = s.iloc[idx + 1]
            if pd.isna(prev_px) or pd.isna(spike_px):
                continue
            cleaned.loc[dt, t] = prev_px
            report.append(
                {
                    "Ticker": t,
                    "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "Bad Price": float(spike_px),
                    "Previous Price": float(prev_px),
                    "Next Price": float(next_px) if pd.notna(next_px) else np.nan,
                }
            )
    return cleaned, report


# =========================
# Strategy calculations
# =========================

@dataclass
class BackfillEvent:
    ticker: str
    needed_date: pd.Timestamp
    used_date: pd.Timestamp
    used_price: float
    context: str


def price_on_or_before(
    prices: pd.DataFrame,
    ticker: str,
    dt: pd.Timestamp,
    backfills: List[BackfillEvent],
    context: str,
) -> float:
    s = prices[ticker].dropna()
    if s.empty:
        raise ValueError(f"No price data for {ticker}")
    dt = pd.Timestamp(dt)
    eligible = s.loc[:dt]
    if len(eligible) == 0:
        first_dt = s.index[0]
        first_px = float(s.iloc[0])
        backfills.append(
            BackfillEvent(
                ticker=ticker,
                needed_date=dt,
                used_date=first_dt,
                used_price=first_px,
                context=context,
            )
        )
        return first_px
    return float(eligible.iloc[-1])


def price_on_or_after(
    prices: pd.DataFrame,
    ticker: str,
    dt: pd.Timestamp,
    lookahead_days: int,
    backfills: List[BackfillEvent],
    context: str,
) -> float:
    """Return the first valid (positive) price on `dt` or within `lookahead_days` after."""
    s = prices[ticker]
    dt = pd.Timestamp(dt)
    end_dt = dt + pd.Timedelta(days=int(lookahead_days))
    window = s.loc[dt:end_dt].dropna()
    if not window.empty:
        window = window[window > 0]
    if window.empty:
        raise ValueError(f"No valid price for {ticker} on/after {dt.date()} within {lookahead_days}d")
    used_dt = pd.Timestamp(window.index[0])
    used_px = float(window.iloc[0])
    if used_dt != dt:
        backfills.append(
            BackfillEvent(
                ticker=ticker,
                needed_date=dt,
                used_date=used_dt,
                used_price=used_px,
                context=context,
            )
        )
    return used_px


def calculate_volatility_asof(prices: pd.DataFrame, asof_dt: pd.Timestamp, window: int = 63) -> pd.Series:
    asof_dt = pd.Timestamp(asof_dt)
    px = prices.loc[:asof_dt].copy()
    rets = px.pct_change()
    if len(rets) >= window:
        vol = rets.rolling(window=window).std().iloc[-1]
    else:
        vol = rets.std()
    return vol


def calculate_inverse_volatility_weights(volatilities: pd.Series) -> pd.Series:
    valid = volatilities[(volatilities > 0) & volatilities.notna()].copy()
    if valid.empty:
        raise ValueError("No valid volatilities to weight.")
    inv = 1.0 / valid
    return inv / inv.sum()


def calculate_momentum_scores_asof(
    prices: pd.DataFrame,
    asof_dt: pd.Timestamp,
    use_calendar_month_end: bool,
    backfills: List[BackfillEvent],
) -> pd.DataFrame:
    asof_dt = pd.Timestamp(asof_dt)
    tickers = list(prices.columns)

    if use_calendar_month_end:
        end_me = pd.Timestamp(asof_dt.date())
        month_ends = pd.date_range(end=end_me, periods=14, freq="ME")
        month_ends = [pd.Timestamp(d.date()) for d in month_ends]
        end_dt = month_ends[-1]
        starts = {"1M": month_ends[-2], "3M": month_ends[-4], "6M": month_ends[-7], "12M": month_ends[-13]}

        out = {}
        for label, start_dt in starts.items():
            rets = {}
            for t in tickers:
                p_end = price_on_or_before(prices, t, end_dt, backfills, context=f"Calendar {label} end")
                p_start = price_on_or_before(prices, t, start_dt, backfills, context=f"Calendar {label} start")
                rets[t] = (p_end / p_start) - 1.0
            out[label] = pd.Series(rets)

        df = pd.DataFrame(out)
        df["Momentum Score"] = df.mean(axis=1)
        return df

    px = prices.loc[:asof_dt].copy()
    out = {}
    lookbacks = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}

    for label, days in lookbacks.items():
        if len(px) == 0:
            out[label] = pd.Series(index=tickers, dtype=float)
            continue

        if len(px) > days:
            p_end = px.iloc[-1]
            p_start = px.iloc[-days]
            out[label] = (p_end / p_start) - 1.0
        else:
            p_end = px.iloc[-1]
            p_start = px.iloc[0]
            for t in tickers:
                backfills.append(
                    BackfillEvent(
                        ticker=t,
                        needed_date=asof_dt - pd.Timedelta(days=days),
                        used_date=px.index[0],
                        used_price=float(p_start[t]) if pd.notna(p_start[t]) else np.nan,
                        context=f"Trading-day {label} start (insufficient history)",
                    )
                )
            out[label] = (p_end / p_start) - 1.0

    df = pd.DataFrame(out)
    df["Momentum Score"] = df.mean(axis=1)
    return df


def calculate_momentum_scores_asof_relative_12_1(
    prices: pd.DataFrame,
    asof_dt: pd.Timestamp,
    use_calendar_month_end: bool,
    backfills: List[BackfillEvent],
) -> pd.DataFrame:
    """
    Compute 12–1 relative momentum (12-month return skipping the most recent month)
    as of `asof_dt`.
    """
    asof_dt = pd.Timestamp(asof_dt)
    tickers = list(prices.columns)

    if prices.empty:
        return pd.DataFrame(index=tickers, data={"Momentum Score": np.nan}, dtype=float)

    if use_calendar_month_end:
        px = prices.loc[:asof_dt].copy()
        if px.empty:
            return pd.DataFrame(index=tickers, data={"Momentum Score": np.nan}, dtype=float)

        me_idx = month_end_trading_days(px.index)
        me_idx = me_idx[me_idx <= asof_dt]
        if len(me_idx) < 13:
            return pd.DataFrame(index=tickers, data={"Momentum Score": np.nan}, dtype=float)

        mom_end_dt = me_idx[-2]
        mom_start_dt = me_idx[-13]

        scores = {}
        for t in tickers:
            try:
                p_start = price_on_or_before(prices, t, mom_start_dt, backfills, context="RelMom 12-1 start (calendar)")
                p_end = price_on_or_before(prices, t, mom_end_dt, backfills, context="RelMom 12-1 end (calendar)")
                scores[t] = (p_end / p_start) - 1.0 if (p_start > 0 and p_end > 0) else np.nan
            except Exception:
                scores[t] = np.nan

        df = pd.DataFrame({"Momentum Score": pd.Series(scores)})
        df = df.reindex(tickers)
        return df

    px = prices.loc[:asof_dt].copy()
    if px.empty:
        return pd.DataFrame(index=tickers, data={"Momentum Score": np.nan}, dtype=float)

    rets = px.pct_change().dropna(how="all")
    lookback_days = 252
    skip_days = 21
    required_len = lookback_days + skip_days
    if len(rets) < required_len:
        return pd.DataFrame(index=tickers, data={"Momentum Score": np.nan}, dtype=float)

    window = rets.iloc[-required_len:-skip_days]
    valid_counts = window.notna().sum(axis=0)
    cum_prod = (1.0 + window).prod(axis=0)
    scores_series = (cum_prod - 1.0).where(valid_counts > 0, np.nan)

    df = pd.DataFrame({"Momentum Score": scores_series})
    df = df.reindex(tickers)
    return df


# =========================
# Portfolio + backtest
# =========================

def compute_portfolio_value(holdings: Dict[str, int], prices_row: pd.Series, cash: float) -> float:
    total = float(cash)
    for t, sh in holdings.items():
        if t in prices_row.index and pd.notna(prices_row[t]):
            total += int(sh) * float(prices_row[t])
    return float(total)


def target_integer_shares(top: List[str], weights: pd.Series, prices_row: pd.Series, total_value: float) -> Dict[str, int]:
    target: Dict[str, int] = {}
    for t in top:
        w = float(weights.get(t, 0.0))
        px = float(prices_row[t])
        if not (px > 0):
            raise ValueError(f"Bad/zero price for {t}.")
        desired_amt = total_value * w
        sh = int(math.floor(desired_amt / px))
        target[t] = max(sh, 0)
    return target


@dataclass
class TradeFill:
    date: pd.Timestamp
    ticker: str
    side: str
    shares: int
    price: float
    notional: float
    cost: float


def rebalance_to_target(
    exec_dt: pd.Timestamp,
    holdings: Dict[str, int],
    cash: float,
    target: Dict[str, int],
    px_row: pd.Series,
    include_costs: bool,
    cost_per_trade: float,
) -> Tuple[Dict[str, int], float, List[TradeFill]]:
    fills: List[TradeFill] = []
    tickers = sorted(set(holdings.keys()) | set(target.keys()))

    # Sells first
    for t in tickers:
        cur = int(holdings.get(t, 0))
        tgt = int(target.get(t, 0))
        diff = tgt - cur
        if diff >= 0:
            continue
        px = float(px_row.get(t, np.nan))
        if not (px > 0):
            continue
        sh = int(-diff)
        notional = sh * px
        cash += notional
        cost = float(cost_per_trade) if include_costs else 0.0
        cash -= cost
        if tgt <= 0:
            holdings.pop(t, None)
        else:
            holdings[t] = tgt
        fills.append(TradeFill(exec_dt, t, "SELL", sh, px, notional, cost))

    # Buys second
    for t in tickers:
        cur = int(holdings.get(t, 0))
        tgt = int(target.get(t, 0))
        diff = tgt - cur
        if diff <= 0:
            continue
        px = float(px_row.get(t, np.nan))
        if not (px > 0):
            continue
        cost = float(cost_per_trade) if include_costs else 0.0
        cash_after_cost = cash - cost
        if cash_after_cost <= 0:
            continue
        desired_sh = int(diff)
        max_affordable = int(math.floor(cash_after_cost / px))
        buy_sh = min(desired_sh, max_affordable)
        if buy_sh <= 0:
            continue
        notional = buy_sh * px
        cash = cash_after_cost - notional
        holdings[t] = cur + buy_sh
        fills.append(TradeFill(exec_dt, t, "BUY", buy_sh, px, notional, cost))

    return holdings, float(cash), fills


def compute_drawdown(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    if eq.empty:
        return equity * np.nan
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return dd.reindex(equity.index)


def calculate_single_equity_metrics(equity: pd.Series) -> Dict:
    """Calculate metrics for a single equity series."""
    eq = equity.dropna()
    if len(eq) < 2:
        return {
            "Annualized Return": np.nan,
            "Annualized Volatility": np.nan,
            "Maximum Drawdown": np.nan,
            "Sharpe (rf=0)": np.nan,
        }

    rets = eq.pct_change().dropna()
    ann = 252.0
    years = len(rets) / ann

    cagr = (float(eq.iloc[-1]) / float(eq.iloc[0])) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = float(rets.std(ddof=1) * math.sqrt(ann)) if rets.std(ddof=1) > 0 else np.nan
    sharpe = (float(rets.mean()) / float(rets.std(ddof=1))) * math.sqrt(ann) if rets.std(ddof=1) > 0 else np.nan
    max_dd = float(compute_drawdown(eq).min())

    return {
        "Annualized Return": cagr,
        "Annualized Volatility": vol,
        "Maximum Drawdown": max_dd,
        "Sharpe (rf=0)": sharpe,
    }


def metrics_from_daily(equity: pd.Series, bench_equity: Optional[pd.Series] = None, benchmark_name: str = "Benchmark") -> pd.DataFrame:
    """Calculate metrics for strategy and optionally benchmark, with alpha/beta."""
    strategy_metrics = calculate_single_equity_metrics(equity)
    strategy_metrics["Name"] = "Rising Assets"
    rows = [strategy_metrics]

    if bench_equity is not None:
        bench_metrics = calculate_single_equity_metrics(bench_equity)
        bench_metrics["Name"] = benchmark_name
        rows.append(bench_metrics)

        eq = equity.dropna()
        bq = bench_equity.dropna().reindex(eq.index).dropna()
        aligned = eq.reindex(bq.index)
        if len(bq) >= 2 and len(aligned) >= 2:
            rs = aligned.pct_change().dropna()
            rb = bq.pct_change().dropna()
            idx = rs.index.intersection(rb.index)
            rs = rs.reindex(idx)
            rb = rb.reindex(idx)
            if len(idx) >= 10 and float(rb.var(ddof=1)) > 0:
                beta = float(rs.cov(rb) / rb.var(ddof=1))
                alpha_ann = float((rs.mean() - beta * rb.mean()) * 252.0)
                rows[0]["Beta vs Benchmark"] = beta
                rows[0]["Annualized Alpha"] = alpha_ann

    df = pd.DataFrame(rows)
    cols = ["Name"] + [c for c in df.columns if c != "Name"]
    return df[cols]


@dataclass
class EquityIssue:
    date: pd.Timestamp
    portfolio_value_raw: float
    portfolio_value_valued: float
    previous_value: float
    drop_pct: float
    missing_prices: str
    holdings_count: int
    note: str


@dataclass
class BacktestResult:
    equity_daily: pd.Series
    equity_benchmark_daily: Optional[pd.Series]
    drawdown_daily: pd.Series
    drawdown_benchmark_daily: Optional[pd.Series]
    trades: pd.DataFrame
    rebalances: pd.DataFrame
    metrics: pd.DataFrame
    spike_report: List[dict]
    currency_report: List[dict]
    backfills: List[BackfillEvent]
    equity_issues: pd.DataFrame
    first_exec_date: pd.Timestamp
    max_mode_info: Tuple[bool, str | None, date | None]
    benchmark_name: str
    component_daily: pd.DataFrame


@st.cache_data(show_spinner=False)
def run_backtest_cached(
    universe: Tuple[str, ...],
    start_iso: str,
    end_iso: str,
    benchmark: str,
    use_calendar_month_end: bool,
    momentum_model: str,
    starting_capital: float,
    include_costs: bool,
    cost_per_trade: float,
    spike_threshold: float,
    valuation_ffill_limit: int,
    guardrail_enabled: bool,
    guardrail_drop_pct: float,
    max_mode_info: Tuple[bool, str | None, str | None],
) -> BacktestResult:

    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    tickers = list(universe)

    if len(tickers) < 5:
        raise ValueError("Universe must contain at least 5 tickers.")
    if end <= start:
        raise ValueError("End date must be after start date.")

    buffer_days = 900
    fetch_start = start - pd.Timedelta(days=buffer_days)
    exec_price_lookahead_days = 3
    fetch_end = end + pd.Timedelta(days=max(10, exec_price_lookahead_days + 3))
    tickers_all = tickers + ([benchmark] if benchmark and benchmark not in tickers else [])

    raw = fetch_price_data_robust_cached(
        tuple(tickers_all),
        fetch_start.date().isoformat(),
        fetch_end.date().isoformat(),
    )

    currency_report: List[dict] = []
    raw, currency_report = fix_gbp_unit_mix_extremes(raw, list(raw.columns))

    cleaned, spike_report = validate_and_clean_prices(raw, threshold=spike_threshold)

    if benchmark:
        if benchmark not in cleaned.columns:
            raise ValueError(f"Benchmark ticker '{benchmark}' not found.")
        bench_px = cleaned[benchmark].copy()
        prices = cleaned.drop(columns=[benchmark])
    else:
        bench_px = None
        prices = cleaned

    prices = prices.dropna(how="all").sort_index()
    if prices.empty:
        raise ValueError("No usable price data.")

    all_month_ends = month_end_trading_days(prices.index)
    exec_month_ends = all_month_ends[(all_month_ends >= start) & (all_month_ends <= end)]
    if len(exec_month_ends) < 2:
        raise ValueError("Not enough month-end trading dates in selected range.")

    first_exec = exec_month_ends[0]
    prior_month_ends = all_month_ends[all_month_ends < first_exec]
    if len(prior_month_ends) == 0:
        raise ValueError("Need at least one prior month-end trading day before start for look-ahead fix.")

    first_signal = prior_month_ends[-1]
    month_ends = pd.DatetimeIndex([first_signal]).append(exec_month_ends)

    prices_val = prices.ffill(limit=int(valuation_ffill_limit)) if valuation_ffill_limit > 0 else prices.copy()

    holdings: Dict[str, int] = {}
    cash = float(starting_capital)

    backfills: List[BackfillEvent] = []
    trade_fills: List[TradeFill] = []
    reb_rows: List[dict] = []

    equity = pd.Series(index=prices.index, dtype=float)
    issues: List[EquityIssue] = []
    prev_equity: Optional[float] = None
    component_rows: Dict[pd.Timestamp, Dict[str, float]] = {}

    if bench_px is not None:
        bench_px = bench_px.dropna().sort_index()

    for i in range(1, len(month_ends)):
        signal_dt = pd.Timestamp(month_ends[i - 1])
        exec_dt = pd.Timestamp(month_ends[i])

        if momentum_model == "Relative 12-1":
            mom_df = calculate_momentum_scores_asof_relative_12_1(prices, signal_dt, use_calendar_month_end, backfills)
        else:
            mom_df = calculate_momentum_scores_asof(prices, signal_dt, use_calendar_month_end, backfills)

        vol = calculate_volatility_asof(prices, signal_dt, window=63)
        valid_mom = mom_df["Momentum Score"].dropna()

        # Valuation prices (limited forward-fill) for portfolio value before trading
        px_row_val_exec = prices_val.loc[:exec_dt].iloc[-1]
        port_val_before = compute_portfolio_value(holdings, px_row_val_exec, cash)

        exec_price_lookahead_days_local = 3
        px_row_exec = prices.loc[:exec_dt].iloc[-1]

        if len(valid_mom) < 5:
            reb_rows.append(
                {
                    "ExecDate": exec_dt,
                    "SignalDate": signal_dt,
                    "Top5": "",
                    "Turnover": 0.0,
                    "PortfolioValueBefore": port_val_before,
                    "PortfolioValueAfter": port_val_before,
                    "CashAfter": cash,
                    "Note": "Skipped (insufficient valid momentum tickers)",
                }
            )
        else:
            top5 = valid_mom.nlargest(5).index.tolist()

            tickers_for_exec = sorted(set(holdings.keys()) | set(top5))
            px_exec: Dict[str, float] = {}

            for t in tickers_for_exec:
                try:
                    px_exec[t] = price_on_or_after(
                        prices,
                        t,
                        exec_dt,
                        lookahead_days=exec_price_lookahead_days_local,
                        backfills=backfills,
                        context=f"Exec price forward-fill (rebalance exec {exec_dt.date()})",
                    )
                except Exception:
                    px_exec[t] = price_on_or_before(
                        prices,
                        t,
                        exec_dt,
                        backfills,
                        context=f"Exec price backward-fill fallback (rebalance exec {exec_dt.date()})",
                    )

            px_row_exec = pd.Series(px_exec, name=exec_dt)
            w = calculate_inverse_volatility_weights(vol.reindex(top5))
            target = target_integer_shares(top5, w, px_row_exec, port_val_before)

            holdings, cash, fills = rebalance_to_target(exec_dt, holdings, cash, target, px_row_exec, include_costs, cost_per_trade)
            trade_fills.extend(fills)

            traded_notional = sum(f.notional for f in fills)
            turnover = traded_notional / port_val_before if port_val_before > 0 else np.nan

            port_val_after = compute_portfolio_value(holdings, px_row_exec, cash)

            reb_rows.append(
                {
                    "ExecDate": exec_dt,
                    "SignalDate": signal_dt,
                    "Top5": ",".join(top5),
                    "Turnover": float(turnover) if pd.notna(turnover) else np.nan,
                    "PortfolioValueBefore": port_val_before,
                    "PortfolioValueAfter": port_val_after,
                    "CashAfter": cash,
                    "Note": "",
                }
            )

        next_exec = pd.Timestamp(month_ends[i + 1]) if (i + 1) < len(month_ends) else None
        seg_end = next_exec if next_exec is not None else (prices.index.max() + pd.Timedelta(days=1))
        seg_idx = prices.index[(prices.index >= exec_dt) & (prices.index < seg_end)]
        if len(seg_idx) == 0:
            continue

        for dt in seg_idx:
            px_raw = prices.loc[dt]
            px_val = prices_val.loc[dt]
            eq_raw = compute_portfolio_value(holdings, px_raw, cash)
            eq_val = compute_portfolio_value(holdings, px_val, cash)

            missing = [t for t in holdings.keys() if pd.isna(px_raw.get(t, np.nan))]
            eq_used = eq_val

            if prev_equity is not None and prev_equity > 0:
                drop_pct = 1.0 - (eq_used / prev_equity)
                if guardrail_enabled and (drop_pct >= guardrail_drop_pct) and (len(missing) > 0):
                    issues.append(
                        EquityIssue(
                            date=pd.Timestamp(dt),
                            portfolio_value_raw=float(eq_raw),
                            portfolio_value_valued=float(eq_val),
                            previous_value=float(prev_equity),
                            drop_pct=float(drop_pct),
                            missing_prices=",".join(sorted(missing))[:500],
                            holdings_count=len(holdings),
                            note="Guardrail applied: large drop with missing prices; kept previous equity.",
                        )
                    )
                    eq_used = float(prev_equity)

            equity.loc[dt] = float(eq_used)
            _crow: Dict[str, float] = {"Cash": float(cash)}
            for _t, _sh in holdings.items():
                _px_t = px_val.get(_t, np.nan)
                _crow[_t] = int(_sh) * float(_px_t) if pd.notna(_px_t) else 0.0
            _crow["Total"] = float(eq_used)
            component_rows[pd.Timestamp(dt)] = _crow
            prev_equity = float(eq_used)

    first_exec_dt = exec_month_ends[0]
    equity = equity.loc[(equity.index >= first_exec_dt) & (equity.index <= exec_month_ends[-1])].dropna()
    if component_rows:
        component_daily = pd.DataFrame(component_rows).T.sort_index()
        component_daily.index.name = "Date"
        component_daily = component_daily.loc[
            (component_daily.index >= first_exec_dt) & (component_daily.index <= exec_month_ends[-1])
        ].fillna(0.0)
    else:
        component_daily = pd.DataFrame()

    bench_equity = None
    if bench_px is not None and not equity.empty:
        b = bench_px.reindex(equity.index).ffill().dropna()
        if not b.empty:
            bench_equity = pd.Series(index=b.index, data=(starting_capital * (b / float(b.iloc[0]))), dtype=float)

    dd = compute_drawdown(equity)
    dd_b = compute_drawdown(bench_equity) if bench_equity is not None else None

    metrics = metrics_from_daily(equity, bench_equity, benchmark_name=benchmark if benchmark else "Benchmark")

    trades_df = pd.DataFrame(
        [
            {"Date": f.date, "Ticker": f.ticker, "Side": f.side, "Shares": f.shares, "Price": f.price, "Notional": f.notional, "Cost": f.cost}
            for f in trade_fills
        ]
    )

    rebalances_df = pd.DataFrame(reb_rows)

    issues_df = pd.DataFrame([i.__dict__ for i in issues]) if issues else pd.DataFrame(
        columns=[
            "date",
            "portfolio_value_raw",
            "portfolio_value_valued",
            "previous_value",
            "drop_pct",
            "missing_prices",
            "holdings_count",
            "note",
        ]
    )

    # Extract max_mode_info for result
    is_max_mode, limiting_symbol, start_date_iso = max_mode_info
    start_date_for_result = date.fromisoformat(start_date_iso) if start_date_iso else None

    return BacktestResult(
        equity_daily=equity,
        equity_benchmark_daily=bench_equity,
        drawdown_daily=dd,
        drawdown_benchmark_daily=dd_b,
        trades=trades_df,
        rebalances=rebalances_df,
        metrics=metrics,
        spike_report=spike_report,
        currency_report=currency_report,
        backfills=backfills,
        equity_issues=issues_df,
        first_exec_date=first_exec_dt,
        max_mode_info=(is_max_mode, limiting_symbol, start_date_for_result),
        benchmark_name=benchmark if benchmark else "Benchmark",
        component_daily=component_daily,
    )


# =========================
# Charts
# =========================

def make_equity_fig(eq: pd.Series, bench: Optional[pd.Series]) -> go.Figure:
    fig = go.Figure()

    # Strategy = blue
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq.values,
            name="Strategy",
            line=dict(width=2, color="#1f77b4"),
        )
    )

    if bench is not None and not bench.empty:
        # Benchmark = red
        fig.add_trace(
            go.Scatter(
                x=bench.index,
                y=bench.values,
                name="Benchmark",
                line=dict(width=2, color="#d62728"),
            )
        )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Equity curve (Growth of £100,000)",
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_drawdown_fig(dd: pd.Series, dd_bench: Optional[pd.Series]) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values * 100.0,
            name="Strategy",
            line=dict(width=2, color="#1f77b4"),
        )
    )

    if dd_bench is not None and not dd_bench.empty:
        fig.add_trace(
            go.Scatter(
                x=dd_bench.index,
                y=dd_bench.values * 100.0,
                name="Benchmark",
                line=dict(width=2, color="#d62728"),
            )
        )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(ticksuffix="%")
    return fig


# =========================
# Excel export
# =========================

def try_plotly_png(fig: go.Figure, scale: float = 2.0) -> Optional[bytes]:
    try:
        return fig.to_image(format="png", scale=scale)
    except Exception:
        return None


def build_excel_bytes(res: BacktestResult, eq_fig: go.Figure, dd_fig: go.Figure) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        metrics = res.metrics.copy()
        metrics.to_excel(writer, sheet_name="Metrics", index=False)

        df_eq = pd.DataFrame({"Equity": res.equity_daily})
        if res.equity_benchmark_daily is not None:
            df_eq["Benchmark"] = res.equity_benchmark_daily
        df_eq.to_excel(writer, sheet_name="DailyEquity", index_label="Date")

        df_dd = pd.DataFrame({"Drawdown": res.drawdown_daily})
        if res.drawdown_benchmark_daily is not None:
            df_dd["Benchmark"] = res.drawdown_benchmark_daily
        df_dd.to_excel(writer, sheet_name="DailyDrawdown", index_label="Date")

        res.rebalances.to_excel(writer, sheet_name="Rebalances", index=False)
        res.trades.to_excel(writer, sheet_name="Trades", index=False)
        res.equity_issues.to_excel(writer, sheet_name="EquityIssues", index=False)

        ws_notes = wb.add_worksheet("Notes")
        ws_notes.write(0, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ws_notes.write(1, 0, "Cash earns 0%.")
        ws_notes.write(2, 0, "Signals computed as-of prior month-end trading day; trades executed at month-end close.")
        if res.currency_report:
            tickers_fixed = sorted({r["Ticker"] for r in res.currency_report})
            ws_notes.write(3, 0, f"GBp/GBP unit-mix extreme corrections applied for: {', '.join(tickers_fixed)}")
        else:
            ws_notes.write(3, 0, "GBp/GBP unit-mix extreme corrections applied: none")

        # Add max mode note if applicable
        is_max_mode, limiting_symbol, start_date_used = res.max_mode_info
        if is_max_mode and limiting_symbol and start_date_used:
            ws_notes.write(4, 0, f"Max date mode: earliest common start date {start_date_used.strftime('%Y-%m-%d')} (limited by {limiting_symbol}).")

        png1 = try_plotly_png(eq_fig)
        png2 = try_plotly_png(dd_fig)

        if png1 is not None or png2 is not None:
            ws_ch = wb.add_worksheet("Charts")
            row = 0
            if png1 is not None:
                ws_ch.insert_image(row, 0, "equity.png", {"image_data": io.BytesIO(png1)})
                row += 22
            if png2 is not None:
                ws_ch.insert_image(row, 0, "drawdown.png", {"image_data": io.BytesIO(png2)})
        else:
            ws_notes.write(6, 0, "Chart images not embedded (install 'kaleido' to enable Plotly PNG export).")

    return buf.getvalue()


# =========================
# Streamlit App
# =========================

def default_universe() -> str:
    return "VUSA.L,EQQQ.L,VUKE.L,VERX.L,IASH.L,VAPX.L,VFEM.L,VJPN.L,IGLS.L,IGLT.L,SGLN.L,XMWX.L,COMX.L"


def app():
    st.set_page_config(page_title="Rising Assets Backtester", layout="wide")
    st.title("Rising Assets — Streamlit Backtester 6.2a")

    today_date = date.today()
    yesterday = today_date - timedelta(days=1)

    with st.sidebar:
        st.header("Inputs")
        universe_text = st.text_area("Universe (comma/newline separated)", value=default_universe(), height=120)
        benchmark = st.text_input("Benchmark ticker", value="^GSPC")

        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input(
                "Start date",
                value=date(2004, 11, 30),
                min_value=date(1990, 1, 1),
                max_value=today_date,
                key="start_date",
            )
        with col2:
            end = st.date_input(
                "End date",
                value=today_date,
                min_value=date(1990, 1, 1),
                max_value=today_date,
                key="end_date",
            )

        use_max_dates = st.checkbox(
            "Use max dates",
            value=False,
            help="Automatically determine the earliest common start date across all tickers in the universe and benchmark. End date will be yesterday.",
        )

        use_calendar_month_end = st.checkbox("Calendar month-end momentum sampling", value=False)
        starting_capital = st.number_input("Starting capital", value=100000.0, step=1000.0)

        momentum_model = st.selectbox(
            "Momentum model",
            ["Alpha Rising Assets", "Relative 12-1"],
            index=0,
        )

        st.subheader("Costs")
        include_costs = st.checkbox("Include transaction costs", value=False)
        cost_per_trade = st.number_input("Cost per trade (£ per ticker traded)", value=5.0, step=1.0)

        st.subheader("Data quality")
        spike_threshold = st.slider(
            "Spike threshold (abs daily change)",
            min_value=0.05,
            max_value=0.80,
            value=0.20,
            step=0.05,
        )

        st.subheader("Valuation guardrail")
        valuation_ffill_limit = st.number_input(
            "Forward-fill limit for valuation (trading days)",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            help="Used only to mark-to-market holdings on days where Yahoo returns missing prices.",
        )

        guardrail_enabled = st.checkbox("Enable guardrail", value=True)
        guardrail_drop_pct = st.slider(
            "Guardrail drop threshold (fraction)",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="If equity drops by >= this fraction in one day AND some held tickers have missing prices, the equity is held flat and logged.",
        )

        run_btn = st.button("Run backtest", type="primary")

    universe = tuple(parse_universe(universe_text))

    if run_btn:
        if len(universe) < 5:
            st.error("Universe must contain at least 5 tickers.")
            st.stop()

        # Handle max dates mode
        is_max_mode = use_max_dates
        limiting_symbol = None
        start_used = start
        end_used = end

        if is_max_mode:
            with st.spinner("Calculating maximum common date range..."):
                benchmark_clean = benchmark.strip()
                all_symbols = list(universe) + ([benchmark_clean] if benchmark_clean else [])
                common_start, limiting_symbol = find_max_common_start_date(all_symbols)
                if common_start is None:
                    st.error("Unable to determine maximum date range for the provided symbols.")
                    st.stop()
                start_used = common_start
                end_used = yesterday
                if end_used <= start_used:
                    st.error("End date must be after start date.")
                    st.stop()

        start_iso = pd.Timestamp(start_used).date().isoformat()
        end_iso = pd.Timestamp(end_used).date().isoformat()

        with st.spinner("Running backtest..."):
            try:
                res = run_backtest_cached(
                    universe=universe,
                    start_iso=start_iso,
                    end_iso=end_iso,
                    benchmark=benchmark.strip(),
                    use_calendar_month_end=use_calendar_month_end,
                    momentum_model=momentum_model,
                    starting_capital=float(starting_capital),
                    include_costs=bool(include_costs),
                    cost_per_trade=float(cost_per_trade),
                    spike_threshold=float(spike_threshold),
                    valuation_ffill_limit=int(valuation_ffill_limit),
                    guardrail_enabled=bool(guardrail_enabled),
                    guardrail_drop_pct=float(guardrail_drop_pct),
                    max_mode_info=(is_max_mode, limiting_symbol, start_iso),
                )
            except ValueError as e:
                st.error(str(e))
                st.stop()

        # Display max mode info if applicable
        is_max_mode_result, limiting_symbol_result, start_date_result = res.max_mode_info
        if is_max_mode_result and limiting_symbol_result and start_date_result:
            st.info(
                f"**Max date mode:** Using earliest common start date **{start_date_result.strftime('%d/%m/%Y')}** "
                f"to **{end_used.strftime('%d/%m/%Y')}**, which is the maximum range available for **{limiting_symbol_result}**."
            )

        st.caption(f"Requested window: {start_iso} to {end_iso} | First execution date: {res.first_exec_date.date().isoformat()}")

        eq_fig = make_equity_fig(res.equity_daily, res.equity_benchmark_daily)
        dd_fig = make_drawdown_fig(res.drawdown_daily, res.drawdown_benchmark_daily)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(eq_fig, use_container_width=True)
        with c2:
            st.plotly_chart(dd_fig, use_container_width=True)

        st.subheader("Metrics (daily returns)")
        if res.metrics is not None and not res.metrics.empty:
            m = res.metrics.copy()
            pct_cols = ["Annualized Return", "Annualized Volatility", "Maximum Drawdown", "Annualized Alpha"]
            for col in pct_cols:
                if col in m.columns:
                    m[col] = (m[col].astype(float) * 100).map("{:.2f}%".format)
            st.dataframe(m, use_container_width=True, hide_index=True)
        else:
            st.info("Metrics table is empty.")

        st.subheader("Rebalances")
        st.dataframe(res.rebalances, use_container_width=True)

        st.subheader("Trades")
        st.dataframe(res.trades, use_container_width=True)

        st.subheader("Equity issues (guardrail log)")
        st.dataframe(res.equity_issues, use_container_width=True)

        excel_bytes = build_excel_bytes(res, eq_fig, dd_fig)
        filename = f"RisingAssets_Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        st.download_button(
            label="Download Excel",
            data=excel_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        if not res.component_daily.empty:
            _csv_bytes = res.component_daily.to_csv().encode("utf-8")
            _csv_filename = f"RisingAssets_ChartData_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                label="Download chart data",
                data=_csv_bytes,
                file_name=_csv_filename,
                mime="text/csv",
            )

        with st.expander("Data quality logs"):
            st.write(f"Price spikes corrected: {len(res.spike_report)}")
            if res.spike_report:
                st.dataframe(pd.DataFrame(res.spike_report), use_container_width=True)

            st.write(f"GBp/GBP unit-mix extreme corrections: {len(res.currency_report)}")
            if res.currency_report:
                st.dataframe(pd.DataFrame(res.currency_report), use_container_width=True)

            st.write(f"Backfills applied: {len(res.backfills)}")
            if res.backfills:
                st.dataframe(pd.DataFrame([e.__dict__ for e in res.backfills]), use_container_width=True)

if __name__ == "__main__":
    app()
