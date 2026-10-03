"""Interactive Streamlit dashboard for the Statistical Arbitrage Lab."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the src-layout package importable when Streamlit Cloud runs app.py
# directly from a repository checkout.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from statarb import PairConfig, adf_diagnostic, backtest_pair, half_life, make_synthetic_pair, performance_report, walk_forward


def parse_price_csv(file) -> pd.DataFrame:
    """Read and validate a CSV with Date, Asset_Y and Asset_X columns."""
    try:
        raw = pd.read_csv(file if hasattr(file, "read") else BytesIO(file))
    except Exception as exc:
        raise ValueError(f"Could not read this CSV file: {exc}") from exc
    required = {"Date", "Asset_Y", "Asset_X"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError("CSV must contain these columns: Date, Asset_Y, Asset_X. Missing: " + ", ".join(missing))
    try:
        dates = pd.to_datetime(raw["Date"], errors="raise")
        values = raw[["Asset_Y", "Asset_X"]].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Date values must be valid dates and asset prices must be numeric: {exc}") from exc
    prices = values.copy()
    prices.index = pd.DatetimeIndex(dates, name="Date")
    prices = prices.sort_index()
    if prices.empty:
        raise ValueError("The CSV contains no price observations.")
    if not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Asset_Y and Asset_X prices must be finite and cannot be blank.")
    if (prices <= 0).any().any():
        raise ValueError("Asset_Y and Asset_X prices must all be positive.")
    if prices.index.has_duplicates:
        raise ValueError("Date values must be unique; combine duplicate dates before uploading.")
    return prices


def performance_cards(returns: pd.Series, fold_results: list[pd.DataFrame]) -> dict[str, float | int]:
    """Summarize only walk-forward out-of-sample returns and their costs."""
    report = performance_report(returns)
    costs = sum(float(frame["transaction_cost"].sum()) for frame in fold_results)
    return {
        "total_return": report["total_return"], "cagr": report["cagr"], "sharpe": report["sharpe"],
        "volatility": report["annualized_volatility"], "max_drawdown": report["max_drawdown"],
        "win_rate": float((returns > 0).mean()) if len(returns) else float("nan"),
        "trading_days": int(len(returns)), "transaction_costs": costs,
    }


def _analysis(prices: pd.DataFrame, settings: dict) -> dict:
    config = PairConfig(
        formation_window=settings["train_window"], z_window=settings["z_lookback"],
        entry_z=settings["entry_z"], exit_z=settings["exit_z"],
        cost_bps=settings["cost_bps"],
    )
    walk = walk_forward(
        prices["Asset_Y"], prices["Asset_X"], train_size=settings["train_window"],
        test_size=settings["test_window"], config=config,
        pvalue_threshold=settings["adf_alpha"], adf_threshold=settings["adf_alpha"],
        max_half_life=settings["max_half_life"],
    )
    full = backtest_pair(prices["Asset_Y"], prices["Asset_X"], config=config)
    if full["beta"].dropna().empty:
        raise ValueError("Not enough observations to estimate a hedge ratio. Increase the data length or reduce the training window.")
    beta = float(full["beta"].dropna().iloc[-1])
    alpha = float(full["intercept"].dropna().iloc[-1])
    train_end = len(prices) - 1
    train_start = max(0, train_end - settings["train_window"])
    training = prices.iloc[train_start:train_end]
    training_spread = np.log(training.Asset_Y) - alpha - beta * np.log(training.Asset_X)
    adf = adf_diagnostic(training_spread, regression="n")
    hl = half_life(training_spread)
    return {
        "walk": walk, "full": full, "cards": performance_cards(walk.out_of_sample_returns, walk.fold_results),
        "diagnostics": {
            "OLS hedge ratio": beta, "Alpha": alpha, "ADF statistic": adf["statistic"],
            "ADF p-value": adf["pvalue"], "Half-life": hl,
            "Regime status": "Tradable regime" if bool(full["regime_ok"].iloc[-1]) else "Filtered / insufficient history",
            "Observations": len(prices),
        },
    }


def _charts(prices: pd.DataFrame, result: dict) -> None:
    full = result["full"]
    walk = result["walk"]
    st.subheader("Market and signal diagnostics")

    normalized = prices / prices.iloc[0] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized.Asset_Y, name="Asset Y"))
    fig.add_trace(go.Scatter(x=normalized.index, y=normalized.Asset_X, name="Asset X"))
    fig.update_layout(title="Normalized prices (start = 100)", yaxis_title="Index", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(x=full.index, y=full.spread, name="Spread", line={"color": "#38bdf8"}))
        fig.update_layout(title="Rolling residual spread", yaxis_title="Log spread", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=full.index, y=full.zscore, name="Z-score", line={"color": "#a78bfa"}))
        fig.add_hline(y=st.session_state.settings["entry_z"], line_dash="dash", line_color="#f87171", annotation_text="Entry")
        fig.add_hline(y=-st.session_state.settings["entry_z"], line_dash="dash", line_color="#f87171")
        exit_z = st.session_state.settings["exit_z"]
        fig.add_hline(y=exit_z, line_dash="dot", line_color="#34d399", annotation_text="Exit")
        fig.add_hline(y=-exit_z, line_dash="dot", line_color="#34d399")
        fig.update_layout(title="Rolling Z-score", yaxis_title="Z-score", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=full.index, y=full.signal, name="Close signal", line_shape="hv"))
    fig.add_trace(go.Scatter(x=full.index, y=full.position, name="Executed position", line_shape="hv"))
    fig.update_layout(title="Signal and next-session position", yaxis_title="Position (-1 / 0 / +1)", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    oos = walk.out_of_sample_returns
    if len(oos):
        equity = (1 + oos).cumprod()
        drawdown = equity / equity.cummax() - 1
        eq_col, dd_col = st.columns(2)
        with eq_col:
            fig = go.Figure(go.Scatter(x=equity.index, y=equity, name="OOS equity", line={"color": "#34d399"}))
            fig.update_layout(title="Walk-forward out-of-sample equity", yaxis_title="Growth of 1.0", hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
        with dd_col:
            fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill="tozeroy", line={"color": "#fb7185"}))
            fig.update_layout(title="Walk-forward out-of-sample drawdown", yaxis_title="Drawdown", hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No walk-forward folds passed the training-only cointegration, ADF, and half-life filters. No out-of-sample performance is available.")


def _format_pct(value: float) -> str:
    return f"{value:.2%}" if np.isfinite(value) else "N/A"


def main() -> None:
    st.set_page_config(page_title="Statistical Arbitrage Lab", page_icon="📈", layout="wide")
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(140deg, #07111f 0%, #0b1728 55%, #101b31 100%); }
    h1 { letter-spacing: -0.04em; }
    [data-testid="stMetric"] { background: rgba(19, 35, 58, .78); border: 1px solid #243957; padding: 16px; border-radius: 14px; }
    </style>
    """, unsafe_allow_html=True)
    st.title("Statistical Arbitrage Lab")
    st.markdown("### Regime-Aware Pairs Trading Research Platform")
    st.caption("Research dashboard · close-based signals · next-session execution · out-of-sample metrics")

    with st.sidebar:
        st.header("Data")
        mode = st.radio("Data mode", ["Synthetic Demo", "CSV Upload"], index=0)
        uploaded = None
        if mode == "Synthetic Demo":
            seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=7, step=1)
            observations = st.slider("Number of observations", min_value=400, max_value=3000, value=1800, step=100)
            st.info("Synthetic data is being used. The demo creates a cointegrated pair from a shared random walk and mean-reverting residual.")
        else:
            uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Required columns: Date, Asset_Y, Asset_X")

        st.header("Strategy controls")
        train_window = st.number_input("Training window", min_value=100, max_value=2000, value=756, step=21)
        test_window = st.number_input("Test window", min_value=20, max_value=504, value=126, step=21)
        z_lookback = st.number_input("Z-score lookback", min_value=10, max_value=504, value=60, step=5)
        entry_z = st.number_input("Entry Z-score", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
        exit_z = st.number_input("Exit Z-score", min_value=0.0, max_value=2.0, value=0.5, step=0.1)
        adf_alpha = st.selectbox("ADF significance level", [0.01, 0.05, 0.10], index=1)
        max_hl = st.number_input("Maximum half-life (bars)", min_value=1.0, max_value=504.0, value=60.0, step=1.0)
        cost_bps = st.number_input("Transaction cost per leg (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        run = st.button("Run Backtest", type="primary", width="stretch")

    settings = {"train_window": int(train_window), "test_window": int(test_window), "z_lookback": int(z_lookback),
                "entry_z": float(entry_z), "exit_z": float(exit_z), "adf_alpha": float(adf_alpha),
                "max_half_life": float(max_hl), "cost_bps": float(cost_bps)}
    st.session_state.settings = settings

    if run:
        try:
            if entry_z <= exit_z:
                raise ValueError("Entry Z-score must be greater than Exit Z-score.")
            if mode == "Synthetic Demo":
                prices = make_synthetic_pair(n=int(observations), seed=int(seed)).rename(columns={"ASSET_A": "Asset_Y", "ASSET_B": "Asset_X"})
            elif uploaded is not None:
                prices = parse_price_csv(uploaded)
            else:
                raise ValueError("Upload a CSV file before running the backtest.")
            if len(prices) <= train_window:
                raise ValueError(f"Need more than {train_window} observations to create a training window and at least one test observation; found {len(prices)}.")
            st.session_state.analysis = _analysis(prices, settings)
            st.session_state.prices = prices
            st.session_state.data_mode = mode
            st.session_state.run_error = None
        except Exception as exc:
            st.session_state.run_error = str(exc)
    if st.session_state.get("run_error"):
        st.error(st.session_state.run_error)

    if "analysis" in st.session_state and not st.session_state.get("run_error"):
        result = st.session_state.analysis
        prices = st.session_state.prices
        if st.session_state.data_mode == "Synthetic Demo":
            st.info("Synthetic data is being used. Reported performance is computed from accepted walk-forward out-of-sample folds only.")
        st.subheader("Performance · walk-forward out-of-sample")
        cards = result["cards"]
        labels = [
            ("Total Return", _format_pct(cards["total_return"])), ("CAGR", _format_pct(cards["cagr"])),
            ("Sharpe Ratio", f"{cards['sharpe']:.2f}" if np.isfinite(cards["sharpe"]) else "N/A"),
            ("Volatility", _format_pct(cards["volatility"])),
            ("Maximum Drawdown", _format_pct(cards["max_drawdown"])),
            ("Win Rate", _format_pct(cards["win_rate"])), ("Trading Days", f"{cards['trading_days']:,}"),
            ("Transaction Costs", f"{cards['transaction_costs']:.4f} return units"),
        ]
        for row in range(0, len(labels), 4):
            cols = st.columns(4)
            for col, (label, value) in zip(cols, labels[row:row + 4]):
                col.metric(label, value)
        st.caption("Win Rate is the share of positive walk-forward return days. Transaction costs are summed in normalized return units across accepted test folds.")

        st.subheader("Diagnostics")
        diag = result["diagnostics"]
        diagnostic_cols = st.columns(7)
        for col, (label, value) in zip(diagnostic_cols, diag.items()):
            if isinstance(value, (int, np.integer)):
                shown = f"{value:,}"
            elif isinstance(value, (float, np.floating)):
                shown = f"{value:.4f}" if np.isfinite(value) else "N/A"
            else:
                shown = str(value)
            col.metric(label, shown)
        st.caption("OLS, ADF, and half-life diagnostics use the latest trailing training window; the current regime reads the latest causal rolling filter state.")

        _charts(prices, result)
        st.subheader("Trade and signal changes")
        full = result["full"]
        changes = full.loc[full.signal.ne(full.signal.shift()), ["zscore", "signal", "position", "beta", "regime_ok"]].copy()
        changes = changes.dropna(subset=["zscore"])
        changes = changes.rename(columns={"zscore": "Z-score", "signal": "Position signal", "position": "Executed position",
                                          "beta": "Hedge ratio", "regime_ok": "Regime"})
        changes.index.name = "Date"
        st.dataframe(changes.reset_index().tail(500), width="stretch", hide_index=True)

        with st.expander("How the Strategy Works", expanded=False):
            st.markdown("""
            - **Engle-Granger cointegration:** each walk-forward training fold checks whether the two log-price series share a stationary linear combination. Only accepted training folds contribute test-period results.
            - **OLS hedge ratio:** ordinary least squares estimates `log(Asset_Y) = alpha + beta × log(Asset_X) + residual` on trailing data. The spread is the residual.
            - **ADF test:** the Augmented Dickey-Fuller test checks for a unit root in the training residual. The selected significance threshold is applied before each test fold.
            - **Half-life:** an AR(1) estimate describes how many bars the residual takes to halve after a deviation. Folds exceeding the selected maximum are skipped.
            - **Rolling Z-score:** the spread is standardized against its trailing rolling mean and standard deviation. Extreme values can open a position; a return toward zero closes it.
            - **Regime filtering:** a causal trailing spread-volatility filter avoids opening or maintaining positions when the spread volatility is above its historical threshold.
            - **Next-day execution:** signals calculated at a bar close are shifted forward one bar before returns are credited, so the signal bar's move is not captured.
            - **Leg-level transaction costs:** costs in basis points are charged separately on each leg's change in notional, including entry, exit, and hedge-weight changes.

            The full-history chart is an illustrative causal signal trace. Performance cards and equity/drawdown charts use only accepted walk-forward out-of-sample test periods.
            """)

        st.warning("This is a research/backtesting tool. Historical or synthetic results do not guarantee future performance.")
        with st.expander("Walk-forward fold diagnostics"):
            st.dataframe(result["walk"].fold_diagnostics, width="stretch", hide_index=True)
    else:
        st.markdown("Choose a data source and strategy controls in the sidebar, then select **Run Backtest** to calculate diagnostics and results.")
        st.warning("This is a research/backtesting tool. Historical or synthetic results do not guarantee future performance.")


if __name__ == "__main__":
    main()
