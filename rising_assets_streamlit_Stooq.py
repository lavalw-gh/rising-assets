"""Rising Assets Strategy — Streamlit Stooq Backtester (v7.2)

Changes in v6.0:
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
- Uses daily Close prices from Stooq (Close).
- Cash earns 0%.

Dependencies
pip install streamlit yfinance pandas numpy xlsxwriter plotly

Optional for embedding PNG charts into Excel:
pip install kaleido
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from urllib.parse import urlencode

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
    """Determine the earliest common start date across symbols using Stooq daily Close.

    We fetch long history (from 1990) per symbol and take each symbol's first valid date.
    The common start date is the latest of those first-valid dates.
    """
    if not symbols:
        return None, None

    early_start = date(1990, 1, 1)
    today_date = date.today()

    first_dates: Dict[str, date] = {}
    for sym in symbols:
        sym = (sym or "").strip()
        if not sym:
            continue
        try:
            s = fetch_stooq_close_cached(sym, early_start.isoformat(), today_date.isoformat())
            s = s.dropna()
            if s.empty:
                continue
            first_dates[sym] = pd.Timestamp(s.index[0]).date()
        except Exception:
            continue

    if not first_dates:
        return None, None

    limiting_symbol = max(first_dates, key=first_dates.get)
    return first_dates[limiting_symbol], limiting_symbol


# =========================
# Data fetch + cleaning
# =========================


STOOQ_BASE_URL = "https://stooq.com/q/d/l/"
STOOQ_INTERVAL = "d"  # always daily
STOOQ_REQUEST_DELAY_SEC = 0.5


def _stooq_symbol_for_request(ticker: str) -> str:
    return (ticker or "").strip().lower()


def _yahoo_ticker_for_currency_lookup(stooq_ticker: str) -> str:
    t = (stooq_ticker or "").strip()
    if t.upper().endswith(".UK"):
        return t[:-3] + ".L"
    return t


@st.cache_data(show_spinner=False)
def get_yahoo_currency_table_cached(stooq_tickers: Tuple[str, ...]) -> pd.DataFrame:
    """Cached currency lookup table via yfinance.

    Rule:
    - GBp (case sensitive) -> pence, convert to GBP by multiplying by 0.01
    - GBP -> already pounds

    Lookup ticker: replace .UK with .L (e.g. SGLN.UK -> SGLN.L).
    """
    rows: List[dict] = []
    for stq in [t for t in stooq_tickers if (t or '').strip()]:
        stq = stq.strip()
        yahoo = _yahoo_ticker_for_currency_lookup(stq)
        currency = ""
        try:
            tobj = yf.Ticker(yahoo)
            try:
                info = getattr(tobj, "info", None)
                if isinstance(info, dict):
                    currency = info.get("currency", "") or ""
            except Exception:
                pass
            if not currency:
                try:
                    fi = getattr(tobj, "fast_info", None)
                    if fi is not None:
                        currency = getattr(fi, "currency", "") or ""
                except Exception:
                    pass
        except Exception:
            currency = ""

        rows.append(
            {
                "Stooq Ticker": stq,
                "Yahoo Lookup": yahoo,
                "Yahoo Currency": currency,
                "PenceToPounds": True if currency == "GBp" else False,
            }
        )
    return pd.DataFrame(rows)


def normalize_prices_to_gbp_from_yahoo_currency(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert Stooq closes into GBP where Yahoo reports currency GBp (pence)."""
    if prices.empty:
        return prices

    tickers = list(prices.columns)
    cur_tbl = get_yahoo_currency_table_cached(tuple(tickers))
    if cur_tbl is None or cur_tbl.empty:
        return prices

    pence_tickers = cur_tbl.loc[cur_tbl["Yahoo Currency"] == "GBp", "Stooq Ticker"].tolist()
    normalized = prices.copy()
    for t in pence_tickers:
        if t in normalized.columns:
            normalized[t] = normalized[t].astype(float) * 0.01
    return normalized


@st.cache_data(show_spinner=False)
def fetch_stooq_close_cached(ticker: str, start_iso: str, end_iso: str) -> pd.Series:
    """Fetch Stooq daily Close series for ticker between start/end (inclusive)."""
    t = (ticker or "").strip()
    if not t:
        raise ValueError("Empty ticker")

    start_dt = pd.Timestamp(start_iso).date()
    end_dt = pd.Timestamp(end_iso).date()

    params = {
        "s": _stooq_symbol_for_request(t),
        "i": STOOQ_INTERVAL,
        "d1": start_dt.strftime("%Y%m%d"),
        "d2": end_dt.strftime("%Y%m%d"),
    }
    url = f"{STOOQ_BASE_URL}?{urlencode(params)}"

    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise ValueError(f"Stooq download failed for '{t}': {e}")
    finally:
        time.sleep(STOOQ_REQUEST_DELAY_SEC)

    if df is None or df.empty:
        raise ValueError(f"No data returned from Stooq for '{t}'.")

    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError(f"Unexpected Stooq schema for '{t}': columns={list(df.columns)}")

    df = df.rename(columns={cols["date"]: "Date", cols["close"]: "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    s = pd.Series(df["Close"].astype(float).values, index=pd.DatetimeIndex(df["Date"]), name=t)
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


@st.cache_data(show_spinner=False)
def fetch_price_data_stooq_cached(tickers: Tuple[str, ...], start_iso: str, end_iso: str) -> pd.DataFrame:
    """Fetch Stooq daily Close prices for tickers (cached)."""
    tlist = [t for t in tickers if (t or '').strip()]
    if not tlist:
        raise ValueError("No tickers provided.")

    series: Dict[str, pd.Series] = {}
    for t in tlist:
        series[t] = fetch_stooq_close_cached(t, start_iso, end_iso)

    prices = pd.DataFrame(series)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(axis=1, how="all")
    if prices.shape[1] == 0:
        raise ValueError("Could not extract price data from any ticker (Stooq).")

    prices = normalize_prices_to_gbp_from_yahoo_currency(prices)
    return prices


def hampel_filter_series(s: pd.Series, window: int = 21, n_sigmas: float = 5.0) -> Tuple[pd.Series, List[dict]]:
    """Hampel filter (trailing window): replace outliers with rolling median."""
    x = s.astype(float).copy()
    if x.dropna().empty:
        return x, []

    med = x.rolling(window=window, min_periods=window).median()
    abs_dev = (x - med).abs()
    mad = abs_dev.rolling(window=window, min_periods=window).median()
    scale = 1.4826 * mad

    outlier = (scale > 0) & scale.notna() & x.notna() & (abs_dev > (n_sigmas * scale))
    outlier = outlier.fillna(False)

    cleaned = x.copy()
    cleaned.loc[outlier] = med.loc[outlier]

    report: List[dict] = []
    for dt in cleaned.index[outlier]:
        report.append(
            {
                "Date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "Bad Price": float(x.loc[dt]) if pd.notna(x.loc[dt]) else np.nan,
                "Replacement Price": float(med.loc[dt]) if pd.notna(med.loc[dt]) else np.nan,
                "Window": int(window),
                "SigmaK": float(n_sigmas),
            }
        )
    return cleaned, report


def apply_hampel_filter_to_prices(prices: pd.DataFrame, window: int = 21, n_sigmas: float = 5.0) -> Tuple[pd.DataFrame, List[dict]]:
    """Apply Hampel filter to every price series and return (cleaned, report)."""
    cleaned = prices.copy().sort_index()
    report: List[dict] = []
    for t in cleaned.columns:
        s_clean, rep = hampel_filter_series(cleaned[t], window=window, n_sigmas=n_sigmas)
        cleaned[t] = s_clean
        for r in rep:
            r["Ticker"] = t
        report.extend(rep)
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
    """Return the first valid (positive) price on `dt` or within `lookahead_days` after.

    This is used to handle cases where Yahoo data is missing on a rebalance execution day
    (e.g., month-end close is NaN for a ticker). If we have to use a later date, we log
    a BackfillEvent so the UI/Excel can show which ticker and which period were affected.
    """
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

    - If `use_calendar_month_end` is True, use calendar month-end anchors:
      momentum = price(month_end[T-2]) / price(month_end[T-13]) - 1,
      where T is the month containing `asof_dt`.
    - Otherwise, use daily returns:
      momentum = product of daily returns over the last 252 trading days,
      excluding the most recent 21 trading days.
    """
    asof_dt = pd.Timestamp(asof_dt)
    tickers = list(prices.columns)

    # If no data at all, return NaNs
    if prices.empty:
        return pd.DataFrame(
            index=tickers,
            data={"Momentum Score": np.nan},
            dtype=float,
        )

    # ------------------------------------------------------------------
    # 1) Calendar month-end version: 12–1 using month-end prices
    # ------------------------------------------------------------------
    if use_calendar_month_end:
        # Use only data up to asof_dt
        px = prices.loc[:asof_dt].copy()
        if px.empty:
            return pd.DataFrame(
                index=tickers,
                data={"Momentum Score": np.nan},
                dtype=float,
            )

        # Month-end trading days up to asof_dt
        me_idx = month_end_trading_days(px.index)
        me_idx = me_idx[me_idx <= asof_dt]

        # Need at least 13 month-ends: we will use
        #   start = month_end[T-13]
        #   end   = month_end[T-2]
        # so we skip the most recent month (T-1 .. T).
        if len(me_idx) < 13:
            return pd.DataFrame(
                index=tickers,
                data={"Momentum Score": np.nan},
                dtype=float,
            )

        # T is the last month-end <= asof_dt
        # Skip the most recent month: use T-2 as end, T-13 as start.
        mom_end_dt = me_idx[-2]
        mom_start_dt = me_idx[-13]

        scores = {}
        for t in tickers:
            try:
                p_start = price_on_or_before(
                    prices,
                    t,
                    mom_start_dt,
                    backfills,
                    context="RelMom 12-1 start (calendar)",
                )
                p_end = price_on_or_before(
                    prices,
                    t,
                    mom_end_dt,
                    backfills,
                    context="RelMom 12-1 end (calendar)",
                )
                if p_start > 0 and p_end > 0:
                    scores[t] = (p_end / p_start) - 1.0
                else:
                    scores[t] = np.nan
            except Exception:
                scores[t] = np.nan

        df = pd.DataFrame({"Momentum Score": pd.Series(scores)})
        # Ensure all universe tickers are present
        df = df.reindex(tickers)
        return df

    # ------------------------------------------------------------------
    # 2) Daily version: 12–1 using daily returns
    # ------------------------------------------------------------------
    # Use data up to asof_dt only
    px = prices.loc[:asof_dt].copy()
    if px.empty:
        return pd.DataFrame(
            index=tickers,
            data={"Momentum Score": np.nan},
            dtype=float,
        )

    # Daily returns
    rets = px.pct_change().dropna(how="all")
    # We approximate:
    #   12 months  ≈ 252 trading days
    #   1 month    ≈ 21 trading days
    lookback_days = 252
    skip_days = 21
    required_len = lookback_days + skip_days

    if len(rets) < required_len:
        # Not enough history yet for a proper 12–1; return NaNs
        return pd.DataFrame(
            index=tickers,
            data={"Momentum Score": np.nan},
            dtype=float,
        )

    # Take the last (252 + 21) daily returns and drop the most recent 21 days:
    # window: rets[t-252-skip .. t-skip-1]
    window = rets.iloc[-required_len:-skip_days]

    # Compute cumulative return over that window per asset: (Π (1+r)) - 1
    # Guard against all-NaN columns.
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
    
    # Calculate strategy metrics
    strategy_metrics = calculate_single_equity_metrics(equity)
    strategy_metrics["Name"] = "Rising Assets"
    
    rows = [strategy_metrics]
    
    # Calculate benchmark metrics if available
    if bench_equity is not None:
        bench_metrics = calculate_single_equity_metrics(bench_equity)
        bench_metrics["Name"] = benchmark_name
        rows.append(bench_metrics)
        
        # Add beta and alpha to strategy row
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
    # Reorder columns to put Name first
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
    backfills: List[BackfillEvent]
    equity_issues: pd.DataFrame
    first_exec_date: pd.Timestamp
    start_shift_note: str
    max_mode_info: Tuple[bool, str | None, date | None]
    benchmark_name: str

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

    raw = fetch_price_data_stooq_cached(
        tuple(tickers_all),
        fetch_start.date().isoformat(),
        fetch_end.date().isoformat(),
    )

    cleaned, spike_report = apply_hampel_filter_to_prices(raw, window=21, n_sigmas=5.0)

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

start_shift_note = ""

if len(prior_month_ends) == 0:
    # No prior month-end is available for the look-ahead fix.
    # Shift the first execution forward by one month-end:
    # signal_dt = exec_month_ends[0], exec_dt = exec_month_ends[1].
    if len(exec_month_ends) < 2:
        raise ValueError("Not enough month-end trading dates in selected range.")

    first_signal = exec_month_ends[0]
    exec_month_ends_adj = exec_month_ends[1:]
    month_ends = pd.DatetimeIndex([first_signal]).append(exec_month_ends_adj)

    start_shift_note = (
        f"Start shifted: no prior month-end before {pd.Timestamp(first_signal).date()} "
        f"for look-ahead fix; first execution is {pd.Timestamp(exec_month_ends_adj[0]).date()}."
    )
else:
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
        # Use valuation prices (limited forward-fill) for portfolio value before trading so
        # missing raw Yahoo prices do not make the portfolio look artificially near-zero.
        px_row_val_exec = prices_val.loc[:exec_dt].iloc[-1]
        port_val_before = compute_portfolio_value(holdings, px_row_val_exec, cash)

        # For execution/trading prices, prefer the exec day price; if it is missing for a
        # ticker, look forward a few days and use the next available valid price (logged
        # to `backfills` so you can see which ticker/period needed replacement).
        # Note: this uses already-fetched Yahoo data; `fetch_end` is extended to include
        # the lookahead window.
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

            # Build execution price row (with forward lookahead for missing prices)
            tickers_for_exec = sorted(set(holdings.keys()) | set(top5))
            px_exec: Dict[str, float] = {}
            for t in tickers_for_exec:
                # Try on/after exec_dt first (if month-end is missing,
                # check a few days after and use the next available price).
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
                    # If no forward price exists (e.g., end of history), fall back
                    # to the last known price on/before exec_dt.
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
            prev_equity = float(eq_used)

    first_exec_dt = exec_month_ends[0]
    equity = equity.loc[(equity.index >= first_exec_dt) & (equity.index <= exec_month_ends[-1])].dropna()

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
        backfills=backfills,
        equity_issues=issues_df,
        first_exec_date=first_exec_dt,
        start_shift_note=start_shift_note,
        max_mode_info=(is_max_mode, limiting_symbol, start_date_for_result),
        benchmark_name=benchmark if benchmark else "Benchmark",
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
        ws_notes.write(3, 0, "Stooq daily Close prices used; Hampel filter applied (k=5, window=21 trading days).")
        if getattr(res, "start_shift_note", ""):
            ws_notes.write(4, 0, res.start_shift_note)
        
        # Add max mode note if applicable
        is_max_mode, limiting_symbol, start_date_used = res.max_mode_info
        if is_max_mode and limiting_symbol and start_date_used:
            ws_notes.write(5, 0, f"Max date mode: earliest common start date {start_date_used.strftime('%Y-%m-%d')} (limited by {limiting_symbol}).")

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
            ws_notes.write(5, 0, "Chart images not embedded (install 'kaleido' to enable Plotly PNG export).")

    return buf.getvalue()

# =========================
# Streamlit App
# =========================

def default_universe() -> str:
    return "VUSA.UK,EQQQ.UK,VUKE.UK,VERX.UK,VAPX.UK,VJPN.UK,VFEM.UK,IUKP.UK,IGLS.UK,IGLT.UK,SLXX.UK,SGLN.UK"

def app():
    st.set_page_config(page_title="Rising Assets Stooq Backtester", layout="wide")
    st.title("Rising Assets Strategy — Streamlit Stooq Backtester (v7.2)")

    today_date = date.today()
    yesterday = today_date - timedelta(days=1)

    with st.sidebar:
        st.header("Inputs")

        universe_text = st.text_area("Universe (comma/newline separated)", value=default_universe(), height=120)
        benchmark = st.text_input("Benchmark ticker", value="^SPX")

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

        use_calendar_month_end = st.checkbox("Calendar month-end momentum sampling", value=True)

        starting_capital = st.number_input("Starting capital", value=100000.0, step=1000.0)

        momentum_model = st.selectbox(
            "Momentum model",
            ["Alpha Rising Assets", "Relative 12-1"],
            index=0,
        )

        st.subheader("Costs")
        include_costs = st.checkbox("Include transaction costs", value=False)
        cost_per_trade = st.number_input("Cost per trade (£ per ticker traded)", value=5.0, step=1.0)

        st.subheader("Valuation guardrail")
        valuation_ffill_limit = st.number_input(
            "Forward-fill limit for valuation (trading days)",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            help="Used only to mark-to-market holdings on days where raw data is missing prices.",
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
                valuation_ffill_limit=int(valuation_ffill_limit),
                guardrail_enabled=bool(guardrail_enabled),
                guardrail_drop_pct=float(guardrail_drop_pct),
                max_mode_info=(is_max_mode, limiting_symbol, start_iso),
            )

        # Display max mode info if applicable
        is_max_mode_result, limiting_symbol_result, start_date_result = res.max_mode_info
        if is_max_mode_result and limiting_symbol_result and start_date_result:
            st.info(
                f"**Max date mode:** Using earliest common start date **{start_date_result.strftime('%d/%m/%Y')}** "
                f"to **{end_used.strftime('%d/%m/%Y')}**, which is the maximum range available for **{limiting_symbol_result}**."
            )

        if getattr(res, "start_shift_note", ""):
            st.info(res.start_shift_note)

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
            # Convert selected columns from fraction to percentage with 2dp
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

        with st.expander("Data quality logs"):
            st.write(f"Hampel filter replacements: {len(res.spike_report)}")
            if res.spike_report:
                st.dataframe(pd.DataFrame(res.spike_report), use_container_width=True)

            st.write(f"Backfills applied: {len(res.backfills)}")
            if res.backfills:
                st.dataframe(pd.DataFrame([e.__dict__ for e in res.backfills]), use_container_width=True)

        st.write("Yahoo currency lookup (used for GBp/pence conversion)")
        try:
            cur_syms = list(universe)
            b = benchmark.strip()
            if b:
                cur_syms.append(b)
            cur_tbl = get_yahoo_currency_table_cached(tuple(cur_syms))
            if cur_tbl is not None and not cur_tbl.empty:
                st.dataframe(cur_tbl, use_container_width=True, hide_index=True)
            else:
                st.info("Currency lookup table is empty.")
        except Exception as e:
            st.info(f"Currency lookup failed: {e}")


if __name__ == "__main__":
    app()
