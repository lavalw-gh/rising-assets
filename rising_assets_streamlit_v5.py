"""Rising Assets Strategy — Streamlit Backtester (v5)

Key features
- Streamlit UI (local or hosted)
- In-memory caching for price downloads and backtest results
- Monthly rebalance strategy with look-ahead fix:
    * Signal computed as-of prior month-end trading day (t-1)
    * Trades executed at month-end trading day close (t)
- Daily equity curve (mark-to-market) for metrics and drawdowns
- Optional transaction costs: £5 per non-zero ticker trade line item
- Excel export (tables + optional embedded PNG charts if Kaleido installed)

Dependencies
    pip install streamlit yfinance pandas numpy xlsxwriter plotly
Optional for PNG chart export into Excel:
    pip install kaleido

Notes
- Uses yfinance with auto_adjust=True and Close prices.
- Cash earns 0%.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

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


def _month_end_trading_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading day for each (year, month) present in index."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if len(idx) == 0:
        return idx
    # Group by month period and take max date
    s = pd.Series(idx, index=idx)
    month_last = s.groupby(idx.to_period("M")).max().sort_values()
    return pd.DatetimeIndex(month_last.values)


def _ensure_datetime(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize() if not isinstance(d, pd.Timestamp) else d.normalize()


# =========================
# Data fetch + cleaning
# =========================

def _normalize_prices_to_gbp(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Normalize prices to GBP: if yfinance reports currency 'GBp' (pence), divide by 100."""
    normalized = prices.copy()
    for ticker in tickers:
        if ticker not in normalized.columns:
            continue
        try:
            tobj = yf.Ticker(ticker)
            currency = ""
            # Try info dict first
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
            if currency == "GBp":
                normalized[ticker] = normalized[ticker] / 100.0
        except Exception:
            # Best-effort only
            pass
    return normalized


@st.cache_data(show_spinner=False)
def fetch_price_data_robust_cached(
    tickers: Tuple[str, ...], start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """Fetch adjusted close prices (Close) using yfinance, with fallback downloads.

    Cached in-memory per Streamlit process.
    """
    tlist = [t for t in tickers if t]
    if not tlist:
        raise ValueError("No tickers provided.")

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

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
            # MultiIndex columns: (ticker, field)
            for t in tlist:
                if hasattr(data.columns, "levels") and t in data.columns.levels[0]:
                    if "Close" in data[t].columns:
                        prices[t] = data[t]["Close"].copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        prices = prices.dropna(axis=1, how="all")
        if prices.shape[1] == 0:
            raise ValueError("Could not extract price data from any ticker.")
        prices = _normalize_prices_to_gbp(prices, list(prices.columns))
        return prices
    except Exception:
        # Fallback to per-ticker download
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
        prices = _normalize_prices_to_gbp(prices, list(prices.columns))
        return prices


def validate_and_clean_prices(prices: pd.DataFrame, threshold: float = 0.20) -> Tuple[pd.DataFrame, List[dict]]:
    """Detect and fix single-day spikes: if |pct_change| and |pct_change_next| exceed threshold, replace with previous."""
    cleaned = prices.copy()
    report: List[dict] = []
    for t in cleaned.columns:
        s = cleaned[t].copy()
        if s.isna().all():
            continue
        pct = s.pct_change()
        pct_next = s.pct_change(-1)
        spikes = (pct.abs() > threshold) & (pct_next.abs() > threshold)
        spike_dates = s[spikes].index
        for dt in spike_dates:
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


def _price_on_or_before(
    prices: pd.DataFrame,
    ticker: str,
    dt: pd.Timestamp,
    backfills: List[BackfillEvent],
    context: str,
) -> float:
    """Price on or before dt; if dt earlier than first price, backfill with first available and log."""
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


def calculate_volatility_asof(prices: pd.DataFrame, asof_dt: pd.Timestamp, window: int = 63) -> pd.Series:
    """Rolling std of daily returns up to asof_dt."""
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
    w = inv / inv.sum()
    return w


def calculate_momentum_scores_asof(
    prices: pd.DataFrame,
    asof_dt: pd.Timestamp,
    use_calendar_month_end: bool,
    backfills: List[BackfillEvent],
) -> pd.DataFrame:
    """Momentum score = mean of trailing 1/3/6/12-month returns."""
    asof_dt = pd.Timestamp(asof_dt)
    tickers = list(prices.columns)

    if use_calendar_month_end:
        # Build calendar month-ends; price sampled on or before these dates
        end_me = pd.Timestamp(asof_dt.date())
        month_ends = pd.date_range(end=end_me, periods=14, freq="ME")
        month_ends = [pd.Timestamp(d.date()) for d in month_ends]
        end_dt = month_ends[-1]
        starts = {
            "1M": month_ends[-2],
            "3M": month_ends[-4],
            "6M": month_ends[-7],
            "12M": month_ends[-13],
        }
        out = {}
        for label, start_dt in starts.items():
            rets = {}
            for t in tickers:
                p_end = _price_on_or_before(prices, t, end_dt, backfills, context=f"Calendar {label} end")
                p_start = _price_on_or_before(prices, t, start_dt, backfills, context=f"Calendar {label} start")
                rets[t] = (p_end / p_start) - 1.0
            out[label] = pd.Series(rets)
        df = pd.DataFrame(out)
        df["Momentum Score"] = df.mean(axis=1)
        return df

    # Trading-day approximations
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


# =========================
# Portfolio + backtest
# =========================

def compute_portfolio_value(holdings: Dict[str, int], prices_row: pd.Series, cash: float) -> float:
    total = float(cash)
    for t, sh in holdings.items():
        if t in prices_row.index and pd.notna(prices_row[t]):
            total += int(sh) * float(prices_row[t])
    return float(total)


def _target_integer_shares(top: List[str], weights: pd.Series, prices_row: pd.Series, total_value: float) -> Dict[str, int]:
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
    side: str  # BUY or SELL
    shares: int
    price: float
    notional: float
    cost: float


def _rebalance_to_target(
    exec_dt: pd.Timestamp,
    holdings: Dict[str, int],
    cash: float,
    target: Dict[str, int],
    px_row: pd.Series,
    include_costs: bool,
    cost_per_trade: float,
) -> Tuple[Dict[str, int], float, List[TradeFill]]:
    """Execute rebalance at exec_dt close: sells then buys, with optional fixed cost per non-zero trade line item."""
    fills: List[TradeFill] = []
    tickers = sorted(set(holdings.keys()) | set(target.keys()))

    # SELL first
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

    # BUY second
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
        # Pay cost only if we actually trade (and only once per ticker)
        cash_after_cost = cash - cost
        if cash_after_cost <= 0:
            # Can't afford even the cost
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


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    eq = equity.dropna()
    if eq.empty:
        return equity * np.nan
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return dd.reindex(equity.index)


def _metrics_from_daily(
    equity: pd.Series,
    bench_equity: Optional[pd.Series] = None,
    rf_daily: float = 0.0,
) -> pd.DataFrame:
    eq = equity.dropna()
    if len(eq) < 2:
        return pd.DataFrame()

    rets = eq.pct_change().dropna()
    n = len(rets)
    ann_factor = 252.0
    years = n / ann_factor

    start_v = float(eq.iloc[0])
    end_v = float(eq.iloc[-1])
    cagr = (end_v / start_v) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    vol = float(rets.std(ddof=1) * math.sqrt(ann_factor)) if rets.std(ddof=1) > 0 else np.nan
    mean_excess = float((rets - rf_daily).mean())
    sharpe = (mean_excess / rets.std(ddof=1)) * math.sqrt(ann_factor) if rets.std(ddof=1) > 0 else np.nan

    dd = _compute_drawdown(eq)
    max_dd = float(dd.min())

    out = {
        "Annualized Return": cagr,
        "Annualized Volatility": vol,
        "Maximum Drawdown": max_dd,
        "Sharpe (rf=0)": sharpe,
    }

    if bench_equity is not None:
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
                alpha_daily = float(rs.mean() - beta * rb.mean())
                alpha_ann = alpha_daily * ann_factor
                out["Beta vs Benchmark"] = beta
                out["Annualized Alpha"] = alpha_ann

    return pd.DataFrame([out])


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


@st.cache_data(show_spinner=False)
def run_backtest_cached(
    universe: Tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
    benchmark: str,
    use_calendar_month_end: bool,
    starting_capital: float,
    include_costs: bool,
    cost_per_trade: float,
    spike_threshold: float,
) -> BacktestResult:
    """Run backtest with daily equity + monthly rebalances. Cached per-process."""

    tickers = list(universe)
    if len(tickers) < 5:
        raise ValueError("Universe must contain at least 5 tickers.")

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end <= start:
        raise ValueError("End date must be after start date.")

    # Fetch buffer to support 12M momentum and the signal (t-1)
    buffer_days = 900
    fetch_start = start - pd.Timedelta(days=buffer_days)
    fetch_end = end + pd.Timedelta(days=10)

    tickers_all = tickers + ([benchmark] if benchmark and benchmark not in tickers else [])

    raw = fetch_price_data_robust_cached(tuple(tickers_all), fetch_start, fetch_end)
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

    # Build month-end trading dates based on available trading days in prices index
    all_month_ends = _month_end_trading_days(prices.index)
    # Ensure we have a signal month end prior to start period
    # We will execute for month ends within [start, end]
    exec_month_ends = all_month_ends[(all_month_ends >= start) & (all_month_ends <= end)]
    if len(exec_month_ends) < 2:
        raise ValueError("Not enough month-end trading dates in selected range.")

    # We need one prior month-end for the first signal
    first_exec = exec_month_ends[0]
    prior_month_ends = all_month_ends[all_month_ends < first_exec]
    if len(prior_month_ends) == 0:
        raise ValueError("Need at least one prior month-end trading day before start for look-ahead fix.")
    first_signal = prior_month_ends[-1]

    month_ends = pd.DatetimeIndex([first_signal]).append(exec_month_ends)

    # Init portfolio (all cash until first execution)
    holdings: Dict[str, int] = {}
    cash = float(starting_capital)
    backfills: List[BackfillEvent] = []

    # Trade logs / rebalance summaries
    trade_fills: List[TradeFill] = []
    reb_rows: List[dict] = []

    # Daily equity curve
    equity = pd.Series(index=prices.index, dtype=float)

    # Benchmark daily equity
    if bench_px is not None:
        bench_px = bench_px.dropna().sort_index()

    # Iterate executions: exec at month_ends[i] for i>=1; signal at month_ends[i-1]
    for i in range(1, len(month_ends)):
        signal_dt = pd.Timestamp(month_ends[i - 1])
        exec_dt = pd.Timestamp(month_ends[i])

        # Compute signal using data up to signal_dt
        mom_df = calculate_momentum_scores_asof(
            prices=prices,
            asof_dt=signal_dt,
            use_calendar_month_end=use_calendar_month_end,
            backfills=backfills,
        )
        vol = calculate_volatility_asof(prices, asof_dt=signal_dt, window=63)
        valid_mom = mom_df["Momentum Score"].dropna()

        # Use execution prices at exec_dt
        px_row = prices.loc[:exec_dt].iloc[-1]
        port_val_before = compute_portfolio_value(holdings, px_row, cash)

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
            w = calculate_inverse_volatility_weights(vol.reindex(top5))
            target = _target_integer_shares(top5, w, px_row, port_val_before)

            # Execute rebalance
            holdings, cash, fills = _rebalance_to_target(
                exec_dt=exec_dt,
                holdings=holdings,
                cash=cash,
                target=target,
                px_row=px_row,
                include_costs=include_costs,
                cost_per_trade=cost_per_trade,
            )
            trade_fills.extend(fills)

            # Turnover
            traded_notional = sum(f.notional for f in fills)
            turnover = traded_notional / port_val_before if port_val_before > 0 else np.nan
            port_val_after = compute_portfolio_value(holdings, px_row, cash)

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

        # Fill daily equity from this exec_dt until next exec_dt (exclusive)
        next_exec = pd.Timestamp(month_ends[i + 1]) if (i + 1) < len(month_ends) else None
        if next_exec is None:
            seg_end = prices.index.max() + pd.Timedelta(days=1)
        else:
            seg_end = next_exec

        seg_idx = prices.index[(prices.index >= exec_dt) & (prices.index < seg_end)]
        if len(seg_idx) == 0:
            continue

        # Mark-to-market for each day
        for dt in seg_idx:
            px_day = prices.loc[dt]
            equity.loc[dt] = compute_portfolio_value(holdings, px_day, cash)

    # Trim equity to requested window (from first exec date to end)
    first_exec_dt = exec_month_ends[0]
    equity = equity.loc[(equity.index >= first_exec_dt) & (equity.index <= exec_month_ends[-1])].dropna()

    bench_equity = None
    if bench_px is not None and not equity.empty:
        b = bench_px.reindex(equity.index).ffill().dropna()
        if not b.empty:
            bench_equity = pd.Series(index=b.index, data=(starting_capital * (b / float(b.iloc[0]))), dtype=float)

    dd = _compute_drawdown(equity)
    dd_b = _compute_drawdown(bench_equity) if bench_equity is not None else None

    metrics = _metrics_from_daily(equity, bench_equity)

    trades_df = pd.DataFrame(
        [
            {
                "Date": f.date,
                "Ticker": f.ticker,
                "Side": f.side,
                "Shares": f.shares,
                "Price": f.price,
                "Notional": f.notional,
                "Cost": f.cost,
            }
            for f in trade_fills
        ]
    )

    rebalances_df = pd.DataFrame(reb_rows)

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
    )


# =========================
# Charts
# =========================

def make_equity_fig(eq: pd.Series, bench: Optional[pd.Series]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy", line=dict(width=2)))
    if bench is not None and not bench.empty:
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="Benchmark", line=dict(width=2)))
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
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values * 100.0, name="Strategy", line=dict(width=2)))
    if dd_bench is not None and not dd_bench.empty:
        fig.add_trace(go.Scatter(x=dd_bench.index, y=dd_bench.values * 100.0, name="Benchmark", line=dict(width=2)))
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

def _try_plotly_png(fig: go.Figure, scale: float = 2.0) -> Optional[bytes]:
    """Return PNG bytes for a Plotly fig if Kaleido is available."""
    try:
        # Plotly handles kaleido internally
        return fig.to_image(format="png", scale=scale)
    except Exception:
        return None


def build_excel_bytes(res: BacktestResult, eq_fig: go.Figure, dd_fig: go.Figure, benchmark: str) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        # Sheets
        metrics = res.metrics.copy()
        if not metrics.empty:
            metrics.insert(0, "Benchmark", benchmark)
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

        # Notes
        ws_notes = wb.add_worksheet("Notes")
        ws_notes.write(0, 0, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        ws_notes.write(1, 0, "Cash earns 0%.")
        ws_notes.write(2, 0, "Signals computed as-of prior month-end trading day; trades executed at month-end close.")

        # Embed images if possible
        png1 = _try_plotly_png(eq_fig)
        png2 = _try_plotly_png(dd_fig)
        if png1 is not None or png2 is not None:
            ws_ch = wb.add_worksheet("Charts")
            row = 0
            if png1 is not None:
                ws_ch.insert_image(row, 0, "equity.png", {"image_data": io.BytesIO(png1)})
                row += 22
            if png2 is not None:
                ws_ch.insert_image(row, 0, "drawdown.png", {"image_data": io.BytesIO(png2)})
        else:
            ws_notes.write(4, 0, "Chart images not embedded (install 'kaleido' to enable Plotly PNG export).")

    return buf.getvalue()


# =========================
# Streamlit App
# =========================

def _default_universe() -> str:
    return "VUSA.L,EQQQ.L,VUKE.L,VERX.L,VAPX.L,VJPN.L,VFEM.L,IUKP.L,IGLS.L,IGLT.L,SLXX.L,SGLN.L"


def app():
    st.set_page_config(page_title="Rising Assets Backtester", layout="wide")
    st.title("Rising Assets — Streamlit Backtester")

    with st.sidebar:
        st.header("Inputs")
        universe_text = st.text_area("Universe (comma/newline separated)", value=_default_universe(), height=120)
        benchmark = st.text_input("Benchmark ticker", value="^GSPC")

        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start (approx)", value=date(2004, 11, 30))
        with col2:
            end = st.date_input("End (approx)", value=date(2018, 12, 31))

        use_calendar_month_end = st.checkbox("Calendar month-end momentum sampling", value=False)

        starting_capital = st.number_input("Starting capital", value=100000.0, step=1000.0)

        st.subheader("Costs")
        include_costs = st.checkbox("Include transaction costs", value=False)
        cost_per_trade = st.number_input("Cost per trade (£ per ticker traded)", value=5.0, step=1.0)

        st.subheader("Data quality")
        spike_threshold = st.slider("Spike threshold (abs daily change)", min_value=0.05, max_value=0.80, value=0.20, step=0.05)

        run_btn = st.button("Run backtest", type="primary")

    universe = tuple(parse_universe(universe_text))

    if run_btn:
        if len(universe) < 5:
            st.error("Universe must contain at least 5 tickers.")
            st.stop()

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        with st.spinner("Running backtest..."):
            res = run_backtest_cached(
                universe=universe,
                start=start_ts,
                end=end_ts,
                benchmark=benchmark.strip(),
                use_calendar_month_end=use_calendar_month_end,
                starting_capital=float(starting_capital),
                include_costs=bool(include_costs),
                cost_per_trade=float(cost_per_trade),
                spike_threshold=float(spike_threshold),
            )

        if res.metrics is None or res.metrics.empty:
            st.warning("Not enough data to compute metrics.")

        # Charts
        eq_fig = make_equity_fig(res.equity_daily, res.equity_benchmark_daily)
        dd_fig = make_drawdown_fig(res.drawdown_daily, res.drawdown_benchmark_daily)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(eq_fig, use_container_width=True)
        with c2:
            st.plotly_chart(dd_fig, use_container_width=True)

        # Metrics
        st.subheader("Metrics (daily returns)")
        if res.metrics is not None and not res.metrics.empty:
            # Format as percentages where appropriate
            m = res.metrics.copy()
            for col in ["Annualized Return", "Annualized Volatility", "Maximum Drawdown", "Annualized Alpha"]:
                if col in m.columns:
                    m[col] = m[col].astype(float)
            st.dataframe(m, use_container_width=True)
        else:
            st.info("Metrics table is empty.")

        # Trades / Rebalances
        st.subheader("Rebalances")
        st.dataframe(res.rebalances, use_container_width=True)

        st.subheader("Trades")
        st.dataframe(res.trades, use_container_width=True)

        # Excel download
        excel_bytes = build_excel_bytes(res, eq_fig, dd_fig, benchmark)
        filename = f"RisingAssets_Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.download_button(
            label="Download Excel",
            data=excel_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Data quality notes
        with st.expander("Data quality logs"):
            st.write(f"Price spikes corrected: {len(res.spike_report)}")
            if res.spike_report:
                st.dataframe(pd.DataFrame(res.spike_report), use_container_width=True)
            st.write(f"Backfills applied: {len(res.backfills)}")
            if res.backfills:
                st.dataframe(pd.DataFrame([e.__dict__ for e in res.backfills]), use_container_width=True)


if __name__ == "__main__":
    app()
