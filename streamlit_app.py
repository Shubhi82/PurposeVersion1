from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression

try:
    from scipy.optimize import nnls as scipy_nnls
except Exception:
    scipy_nnls = None

from data_processing import (
    build_application_series,
    build_channel_comparison_table,
    build_product_spend_series,
    build_tactic_time_series,
    exclude_direct_mail_rows,
    filter_rolled_up_products,
    load_marketing_spend_data,
    load_marketing_spend_raw,
    load_dm_data,
    load_modeling_data,
    load_originations_data,
    load_originations_raw,
    ms_spend_by_channel,
    ms_spend_by_product,
    ms_spend_by_state,
    ms_spend_by_state_tactic,
    ms_spend_by_tactic,
    ms_spend_over_time,
    ms_tactic_channel_matrix,
    od_channel_state_matrix,
    od_funnel_by_state,
    od_metrics_by_channel,
    od_metrics_by_product,
    od_metrics_by_state,
    od_metrics_over_time,
    prepare_modeling_dataset,
    prepare_raw_mmm_dataset,
    summarize_dm_data,
)
from modeling import fit_channel_models
from utils import (
    CHANNELS,
    DEFAULT_DATA_PATH,
    DM_DATA_PATH,
    MARKETING_SPEND_PATH,
    ORIGINATIONS_PATH,
    OUTCOME_COLUMNS,
    PRODUCT_ALL_LABEL,
    TACTIC_COLUMNS,
    TIME_GRAINS,
    expand_rollup_product,
    format_metric,
    get_available_products,
    get_available_rolled_up_products,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Marketing Analytics and Modeling", layout="wide")

PALETTE = px.colors.qualitative.Set2

# ---------------------------------------------------------------------------
# V2 MMM Constants
# ---------------------------------------------------------------------------
WEIGHT_SCHEMES = {
    "lag2_peak  [0.10, 0.25, 0.40, 0.20, 0.05]  ← locked best": [0.10, 0.25, 0.40, 0.20, 0.05],
    "lag1_peak  [0.15, 0.40, 0.30, 0.10, 0.05]":                 [0.15, 0.40, 0.30, 0.10, 0.05],
    "immediate  [0.70, 0.20, 0.07, 0.02, 0.01]":                 [0.70, 0.20, 0.07, 0.02, 0.01],
    "slow_decay [0.30, 0.30, 0.20, 0.15, 0.05]":                 [0.30, 0.30, 0.20, 0.15, 0.05],
    "uniform    [0.20, 0.20, 0.20, 0.20, 0.20]":                 [0.20, 0.20, 0.20, 0.20, 0.20],
}
TRAIN_YEARS_V2   = [2024, 2025]
TRAIN_YEARS_V3   = [2024, 2025]
TEST_YEARS_V3    = [2026]
ADSTOCK_DECAY    = 0.5
FOURIER_K        = 2
PRESCREEN_COL    = "Prescreen"
ADSTOCK_COLS     = ["DSP", "LeadGen", "Paid Search", "Paid Social", "Referrals", "Sweepstakes"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def solve_nonnegative_least_squares(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, str]:
    """Fit a non-negative linear model, preferring SciPy NNLS when available."""
    if scipy_nnls is not None:
        coefs, _ = scipy_nnls(X, y)
        return coefs.astype(float), "scipy.nnls"

    model = LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y)
    return np.asarray(model.coef_, dtype=float), "sklearn.LinearRegression(positive=True)"

def chart_with_table(fig, table_df: pd.DataFrame, table_label: str = "View data") -> None:
    st.plotly_chart(fig, use_container_width=True)
    with st.expander(f"📋 {table_label}"):
        st.dataframe(table_df, use_container_width=True, hide_index=True)

def apply_bottom_legend(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    )
    return fig

def bar_h(df, x, y, title, color=None):
    fig = px.bar(df, x=x, y=y, orientation="h", title=title,
                 color=color, color_discrete_sequence=PALETTE, text_auto=".3s")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def bar_v(df, x, y, title, barmode="group"):
    if isinstance(y, list):
        fig = px.bar(df, x=x, y=y, title=title, barmode=barmode,
                     color_discrete_sequence=PALETTE, text_auto=".3s")
    else:
        fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=PALETTE, text_auto=".3s")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def line_chart(df, x, y, title):
    fig = px.line(df, x=x, y=y, title=title, color_discrete_sequence=PALETTE, markers=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def heatmap_fig(df, title):
    idx_col = df.columns[0]
    mat = df.set_index(idx_col)
    fig = go.Figure(data=go.Heatmap(
        z=mat.values, x=mat.columns.tolist(), y=mat.index.tolist(),
        colorscale="Blues",
        text=[[f"${v:,.0f}" for v in row] for row in mat.values],
        texttemplate="%{text}",
        hovertemplate="%{y} | %{x}: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading marketing spend data…")
def auto_load_marketing_spend():
    return load_marketing_spend_data(MARKETING_SPEND_PATH, fallback_source=DEFAULT_DATA_PATH)

@st.cache_data(show_spinner="Loading originations data…")
def auto_load_originations():
    return load_originations_data(ORIGINATIONS_PATH, fallback_source=DEFAULT_DATA_PATH)

@st.cache_data(show_spinner="Loading modeling workbook…")
def auto_load_modeling(path: str, time_grain: str = "Weekly"):
    return load_modeling_data(Path(path), time_grain=time_grain)

@st.cache_data(show_spinner=False)
def cached_load_data(source: bytes, time_grain: str = "Weekly"):
    return load_modeling_data(source, time_grain=time_grain)

@st.cache_data(show_spinner="Preparing raw all-state MMM dataset…")
def auto_prepare_v3_dataset(time_grain: str = "Weekly") -> pd.DataFrame:
    marketing = load_marketing_spend_data(MARKETING_SPEND_PATH, fallback_source=DEFAULT_DATA_PATH)
    originations = load_originations_data(ORIGINATIONS_PATH, fallback_source=DEFAULT_DATA_PATH)
    return prepare_raw_mmm_dataset(marketing, originations, time_grain=time_grain)


@st.cache_data(show_spinner="Preparing V4 rolled-up dataset…")
def auto_prepare_v4_dataset(time_grain: str = "Weekly") -> pd.DataFrame:
    marketing = load_marketing_spend_data(MARKETING_SPEND_PATH, fallback_source=DEFAULT_DATA_PATH)
    originations = load_originations_data(ORIGINATIONS_PATH, fallback_source=DEFAULT_DATA_PATH)
    marketing = filter_rolled_up_products(exclude_direct_mail_rows(marketing))
    originations = filter_rolled_up_products(originations)
    return prepare_raw_mmm_dataset(marketing, originations, time_grain=time_grain)


@st.cache_data(show_spinner="Loading Direct Mail data…")
def auto_load_dm_data() -> pd.DataFrame:
    return load_dm_data(DM_DATA_PATH)

# ---------------------------------------------------------------------------
# V2 MMM transformation helpers
# ---------------------------------------------------------------------------

def _circular_spread(series: np.ndarray, weights: list) -> np.ndarray:
    """Distribute a spend drop across future weeks using provided weights."""
    out = np.zeros(len(series))
    for i, val in enumerate(series):
        if val > 0:
            for lag, w in enumerate(weights):
                if i + lag < len(series):
                    out[i + lag] += val * w
    return out

def _adstock(series: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """Geometric adstock — carry forward `decay` fraction each week."""
    r = np.zeros(len(series))
    r[0] = series[0]
    for i in range(1, len(series)):
        r[i] = series[i] + decay * r[i - 1]
    return r

def _hill(x: np.ndarray) -> np.ndarray:
    """Hill saturation — diminishing returns. Returns values in [0,1)."""
    pos = x[x > 0]
    gamma = float(np.percentile(pos, 50)) if len(pos) > 0 else 1.0
    return x / (x + max(gamma, 1e-9))

def _durbin_watson(resid: np.ndarray) -> float:
    d = np.diff(resid)
    return float(np.dot(d, d) / max(np.dot(resid, resid), 1e-12))

def _fourier_features(iso_week_series: pd.Series, k: int = 2) -> pd.DataFrame:
    """Generate k sin+cos Fourier pairs for a weekly seasonal cycle (P=52)."""
    P = 52.0
    wk = ((iso_week_series - 1) % 52) + 1
    feats = {}
    for i in range(1, k + 1):
        feats[f"sin_{i}"] = np.sin(2 * np.pi * i * wk / P)
        feats[f"cos_{i}"] = np.cos(2 * np.pi * i * wk / P)
    return pd.DataFrame(feats, index=iso_week_series.index)


def get_v3_time_settings(time_grain: str) -> dict[str, object]:
    if time_grain in {"Fortnight", "Fortnightly"}:
        return {
            "period_col": "FORTNIGHT",
            "seasonal_features": [f"BW_{period}" for period in range(2, 27)],
            "seasonality_label": "fortnight dummies (BW_2...BW_26)",
            "test_unit_label": "fortnights",
        }
    return {
        "period_col": "ISO_WEEK",
        "seasonal_features": [f"W_{period}" for period in range(2, 53)],
        "seasonality_label": "week dummies (W_2...W_52)",
        "test_unit_label": "weeks",
    }

def transform_for_mmm(df: pd.DataFrame, weights: list, include_fourier: bool = True) -> pd.DataFrame:
    """
    Apply all V2 transformations in order:
      4. Prescreen circular spread
      5. Adstock on digital tactics
      6. Hill saturation
      7. Fourier seasonality
    Returns df with new _final columns appended.
    """
    g = df.copy().reset_index(drop=True)

    # Step 7 — Fourier (needs ISO_WEEK)
    if include_fourier and "ISO_WEEK" in g.columns:
        fourier = _fourier_features(g["ISO_WEEK"], k=FOURIER_K)
        g = pd.concat([g, fourier], axis=1)

    # Step 4 — Prescreen spread
    if PRESCREEN_COL in g.columns:
        g[f"{PRESCREEN_COL}_final"] = _circular_spread(
            g[PRESCREEN_COL].fillna(0).values, weights
        )

    # Steps 5+6 — Adstock then Hill for digital tactics
    for col in ADSTOCK_COLS:
        if col in g.columns:
            ads = _adstock(g[col].fillna(0).values, ADSTOCK_DECAY)
            g[f"{col}_final"] = _hill(ads)

    return g


def build_mmm_feature_cols(
    df: pd.DataFrame,
    include_fourier: bool = True,
    time_grain: str = "Weekly",
) -> list[str]:
    seasonal = []
    if include_fourier:
        seasonal = [f"sin_{k}" for k in range(1, FOURIER_K + 1)] + \
                   [f"cos_{k}" for k in range(1, FOURIER_K + 1)]
    else:
        # Drop the first period as baseline to avoid perfect multicollinearity.
        seasonal = get_v3_time_settings(time_grain)["seasonal_features"]
    prescreen = [f"{PRESCREEN_COL}_final"]
    digital   = [f"{c}_final" for c in ADSTOCK_COLS]
    all_feats = seasonal + prescreen + digital
    return [c for c in all_feats if c in df.columns]


def run_mmm_for_channel(
    df: pd.DataFrame,
    channel: str,
    weights: list,
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
    include_fourier: bool = True,
) -> dict | None:
    """
    Run the full V2 MMM pipeline for one channel.
    Returns a result dict or None if insufficient data.
    """
    train_years = TRAIN_YEARS_V2 if train_years is None else train_years

    ch_df = df[df["CHANNEL_CD"] == channel].copy()
    if ch_df.empty:
        return None

    ch_df = ch_df.sort_values(["ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)
    ch_df = transform_for_mmm(ch_df, weights, include_fourier=include_fourier)

    train = ch_df[ch_df["ISO_YEAR"].isin(train_years)].copy()
    if test_years is None:
        test = ch_df[~ch_df["ISO_YEAR"].isin(train_years)].copy()
    else:
        test = ch_df[ch_df["ISO_YEAR"].isin(test_years)].copy()

    if len(train) < 5 or test.empty:
        return None

    feat_cols = build_mmm_feature_cols(ch_df, include_fourier=include_fourier)
    if not feat_cols:
        return None

    X_tr = np.column_stack([np.ones(len(train)), train[feat_cols].fillna(0).values])
    X_te = np.column_stack([np.ones(len(test)),  test[feat_cols].fillna(0).values])
    y_tr = train["APPLICATIONS"].fillna(0).values.astype(float)
    y_te = test["APPLICATIONS"].fillna(0).values.astype(float)

    # Step 9 — NNLS / positive fallback
    coefs, solver_name = solve_nonnegative_least_squares(X_tr, y_tr)
    yhat_tr  = np.clip(X_tr @ coefs, 0, None)
    yhat_te  = np.clip(X_te @ coefs, 0, None)

    # Step 10 — Diagnostics
    def _r2(y, yh):
        ss_res = np.sum((y - yh) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / max(ss_tot, 1e-12)

    def _mae(y, yh): return float(np.mean(np.abs(y - yh)))

    def _mape(y, yh):
        mask = y != 0
        return float(np.mean(np.abs((y[mask] - yh[mask]) / y[mask])) * 100) if mask.any() else np.nan

    n, p = len(y_tr), len(feat_cols)
    r2_tr = _r2(y_tr, yhat_tr)
    adj_r2 = 1 - (1 - r2_tr) * (n - 1) / max(n - p - 1, 1)

    # Step 11 — Contribution decomp (test period)
    col_names = ["const"] + feat_cols
    contrib = pd.DataFrame(
        {c: X_te[:, i] * coefs[i] for i, c in enumerate(col_names)},
        index=test.index,
    )
    contrib.insert(0, "period_label",
                   test["ISO_YEAR"].astype(str) + "-W" + test["ISO_WEEK"].astype(str).str.zfill(2))
    contrib.insert(1, "Actual",    y_te)
    contrib.insert(2, "Predicted", yhat_te)

    # Fitted frame (full)
    modeled_df = ch_df[ch_df["ISO_YEAR"].isin(train_years + ([] if test_years is None else test_years))].copy() \
        if test_years is not None else ch_df.copy()
    full_periods = (
        modeled_df["ISO_YEAR"].astype(str) + "-W" + modeled_df["ISO_WEEK"].astype(str).str.zfill(2)
    )
    X_full   = np.column_stack([np.ones(len(modeled_df)), modeled_df[feat_cols].fillna(0).values])
    yhat_all = np.clip(X_full @ coefs, 0, None)
    fitted = pd.DataFrame({
        "period_label": full_periods,
        "APPLICATIONS": modeled_df["APPLICATIONS"].fillna(0).values,
        "Predicted":    yhat_all,
        "split": [
            "Train" if yr in train_years else "Test"
            for yr in modeled_df["ISO_YEAR"]
        ],
    })

    coef_df = pd.DataFrame({
        "feature":     col_names,
        "coefficient": coefs,
    })

    return {
        "channel":    channel,
        "fitted":     fitted,
        "coef_df":    coef_df,
        "contrib":    contrib,
        "train_r2":   round(r2_tr, 4),
        "adj_r2":     round(adj_r2, 4),
        "test_r2":    round(_r2(y_te, yhat_te), 4),
        "train_mae":  round(_mae(y_tr, yhat_tr), 1),
        "test_mae":   round(_mae(y_te, yhat_te), 1),
        "train_mape": round(_mape(y_tr, yhat_tr), 2),
        "test_mape":  round(_mape(y_te, yhat_te), 2),
        "dw":         round(_durbin_watson(y_tr - yhat_tr), 3),
        "solver":     solver_name,
        "n_train":    len(train),
        "n_test":     len(test),
        "avg_test_apps": float(np.mean(y_te)) if len(y_te) > 0 else 0,
    }


def _r2_score(y: np.ndarray, yh: np.ndarray) -> float:
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-12)


def _mae_score(y: np.ndarray, yh: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yh)))


def _mape_score(y: np.ndarray, yh: np.ndarray) -> float:
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yh[mask]) / y[mask])) * 100) if mask.any() else np.nan


def _summarize_mmm_fit(frame: pd.DataFrame, split_label: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["period_label", "APPLICATIONS", "Predicted", "split"])
    summary = (
        frame.groupby("period_label", dropna=False)[["APPLICATIONS", "Predicted"]]
        .sum()
        .reset_index()
    )
    summary["split"] = split_label
    return summary


def run_mmm_rolling_v3(
    df: pd.DataFrame,
    channel: str,
    weights: list,
    time_grain: str = "Weekly",
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
) -> dict | None:
    """Run V3 using a fixed baseline fit plus rolling 2026 period-by-period prediction."""
    train_years = TRAIN_YEARS_V3 if train_years is None else train_years
    test_years = TEST_YEARS_V3 if test_years is None else test_years
    settings = get_v3_time_settings(time_grain)
    period_col = settings["period_col"]

    ch_df = df[df["CHANNEL_CD"] == channel].copy()
    if ch_df.empty or period_col not in ch_df.columns:
        return None

    ch_df = ch_df.sort_values(["ISO_YEAR", period_col]).reset_index(drop=True)
    ch_df = transform_for_mmm(ch_df, weights, include_fourier=False)
    ch_df = ch_df[ch_df["ISO_YEAR"].isin(train_years + test_years)].copy()

    train = ch_df[ch_df["ISO_YEAR"].isin(train_years)].copy()
    test = ch_df[ch_df["ISO_YEAR"].isin(test_years)].copy()
    if len(train) < 5 or test.empty:
        return None

    feat_cols = build_mmm_feature_cols(ch_df, include_fourier=False, time_grain=time_grain)
    if not feat_cols:
        return None

    X_train = np.column_stack([np.ones(len(train)), train[feat_cols].fillna(0).values])
    y_train = train["APPLICATIONS"].fillna(0).values.astype(float)
    baseline_coefs, baseline_solver = solve_nonnegative_least_squares(X_train, y_train)
    train["Predicted"] = np.clip(X_train @ baseline_coefs, 0, None)

    train_period_fit = _summarize_mmm_fit(train[["period_label", "APPLICATIONS", "Predicted"]], "Train (baseline)")
    coef_df = pd.DataFrame({
        "feature": ["const"] + feat_cols,
        "coefficient": baseline_coefs,
    })

    test_keys = (
        test[["ISO_YEAR", period_col, "period_label"]]
        .drop_duplicates()
        .sort_values(["ISO_YEAR", period_col])
        .reset_index(drop=True)
    )
    rolling_frames: list[pd.DataFrame] = []
    rolling_detail_rows: list[dict[str, object]] = []

    for key in test_keys.itertuples(index=False):
        year = int(key.ISO_YEAR)
        period_value = int(getattr(key, period_col))
        period_label = str(key.period_label)

        prior_test_mask = (
            (ch_df["ISO_YEAR"].isin(test_years))
            & (
                (ch_df["ISO_YEAR"] < year)
                | ((ch_df["ISO_YEAR"] == year) & (ch_df[period_col] < period_value))
            )
        )
        step_train = ch_df[ch_df["ISO_YEAR"].isin(train_years) | prior_test_mask].copy()
        period_rows = ch_df[(ch_df["ISO_YEAR"] == year) & (ch_df[period_col] == period_value)].copy()
        if len(step_train) < 5 or period_rows.empty:
            continue

        X_step_train = np.column_stack([np.ones(len(step_train)), step_train[feat_cols].fillna(0).values])
        y_step_train = step_train["APPLICATIONS"].fillna(0).values.astype(float)
        step_coefs, step_solver = solve_nonnegative_least_squares(X_step_train, y_step_train)

        X_period = np.column_stack([np.ones(len(period_rows)), period_rows[feat_cols].fillna(0).values])
        period_rows["Predicted"] = np.clip(X_period @ step_coefs, 0, None)
        rolling_frames.append(period_rows[["period_label", "APPLICATIONS", "Predicted"]].copy())

        rolling_detail_rows.append({
            "period_label": period_label,
            "Actual": float(period_rows["APPLICATIONS"].sum()),
            "Predicted": float(period_rows["Predicted"].sum()),
            "Train Rows": int(len(step_train)),
            "Predicted Rows": int(len(period_rows)),
            "Solver": step_solver,
        })

    if not rolling_frames:
        return None

    rolling_raw = pd.concat(rolling_frames, ignore_index=True)
    rolling_fit = _summarize_mmm_fit(rolling_raw, "Test (rolling)")
    fitted = pd.concat([train_period_fit, rolling_fit], ignore_index=True)

    rolling_detail = pd.DataFrame(rolling_detail_rows)
    y_test = rolling_detail["Actual"].to_numpy(dtype=float)
    yhat_test = rolling_detail["Predicted"].to_numpy(dtype=float)
    yhat_train = train["Predicted"].to_numpy(dtype=float)

    n_train, p = len(y_train), len(feat_cols)
    train_r2 = _r2_score(y_train, yhat_train)
    adj_r2 = 1 - (1 - train_r2) * (n_train - 1) / max(n_train - p - 1, 1)

    return {
        "channel": channel,
        "time_grain": time_grain,
        "fitted": fitted,
        "coef_df": coef_df,
        "rolling_detail": rolling_detail,
        "train_r2": round(train_r2, 4),
        "adj_r2": round(adj_r2, 4),
        "test_r2": round(_r2_score(y_test, yhat_test), 4),
        "train_mape": round(_mape_score(y_train, yhat_train), 2),
        "test_mape": round(_mape_score(y_test, yhat_test), 2),
        "train_mae": round(_mae_score(y_train, yhat_train), 1),
        "test_mae": round(_mae_score(y_test, yhat_test), 1),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "solver": baseline_solver,
        "avg_test_apps": float(np.mean(y_test)) if len(y_test) > 0 else 0,
    }


def render_mmm_rolling_result(result: dict, avg_apps: float, key_prefix: str) -> None:
    """Render rolling V3 results with train baseline and expanding-window test prediction."""
    mae_pct = result["test_mae"] / max(avg_apps, 1) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Train R²", f"{result['train_r2']:.3f}")
    c2.metric("Rolling Test R²", f"{result['test_r2']:.3f}")
    c3.metric("Rolling Test MAPE", f"{result['test_mape']:.2f}%")
    c4.metric("Rolling Test MAE %", f"{mae_pct:.1f}%")
    c5.metric("Solver", result["solver"])

    fitted = result["fitted"]
    fig = go.Figure()
    for split, color in [("Train (baseline)", "#378ADD"), ("Test (rolling)", "#E24B4A")]:
        sub = fitted[fitted["split"] == split]
        fig.add_trace(go.Scatter(
            x=sub["period_label"],
            y=sub["APPLICATIONS"],
            mode="lines+markers",
            name=f"Actual ({split})",
            line=dict(color=color, width=1.5),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=sub["period_label"],
            y=sub["Predicted"],
            mode="lines",
            name=f"Predicted ({split})",
            line=dict(color=color, width=1.5, dash="dash"),
        ))
    fig.update_layout(
        title="Actual vs Predicted — baseline train + rolling 2026 test",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}-actual-vs-predicted")

    col_left, col_right = st.columns(2)

    with col_left:
        coef_sub = result["coef_df"][result["coef_df"]["feature"].str.endswith("_final")].copy()
        coef_sub = coef_sub.loc[coef_sub["coefficient"] > 0].copy()
        if not coef_sub.empty:
            coef_sub["label"] = coef_sub["feature"].str.replace("_final", "", regex=False)
            fig_coef = px.bar(
                coef_sub.sort_values("coefficient"),
                x="coefficient",
                y="label",
                orientation="h",
                title="Baseline tactic coefficients",
                color_discrete_sequence=["#378ADD"],
            )
            fig_coef.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_coef, use_container_width=True, key=f"{key_prefix}-coef-bar")

    with col_right:
        size_df = result["rolling_detail"][["period_label", "Train Rows"]].copy()
        fig_size = px.bar(
            size_df,
            x="period_label",
            y="Train Rows",
            title="Train set size used for each rolling prediction",
            color_discrete_sequence=["#185FA5"],
        )
        fig_size.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_size, use_container_width=True, key=f"{key_prefix}-train-size")

    with st.expander("📋 Rolling prediction detail"):
        st.dataframe(result["rolling_detail"], use_container_width=True, hide_index=True)

    with st.expander("📋 Baseline coefficient table"):
        st.dataframe(result["coef_df"], use_container_width=True, hide_index=True)


def render_mmm_channel_result(result: dict, avg_apps: float) -> None:
    """Render all V2 outputs for a single channel."""
    mae_pct = result["test_mae"] / max(avg_apps, 1) * 100
    dw_ok   = 1.5 <= result["dw"] <= 2.5

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Train R²",   f"{result['train_r2']:.3f}")
    c2.metric("Test R²",    f"{result['test_r2']:.3f}")
    c3.metric("Test MAE %", f"{mae_pct:.1f}%",
              delta="good" if mae_pct < 15 else ("ok" if mae_pct < 25 else "needs work"),
              delta_color="normal" if mae_pct < 15 else "inverse")
    c4.metric("Adj R²",     f"{result['adj_r2']:.3f}")
    c5.metric("DW",         f"{result['dw']:.2f}",
              delta="✓ clean" if dw_ok else "⚠ check",
              delta_color="normal" if dw_ok else "inverse")

    # Actual vs Predicted
    fitted = result["fitted"]
    fig = go.Figure()
    for split, color in [("Train", "#378ADD"), ("Test", "#E24B4A")]:
        sub = fitted[fitted["split"] == split]
        fig.add_trace(go.Scatter(x=sub["period_label"], y=sub["APPLICATIONS"],
                                 mode="lines+markers", name=f"Actual ({split})",
                                 line=dict(color=color, width=1.5), marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=sub["period_label"], y=sub["Predicted"],
                                 mode="lines", name=f"Predicted ({split})",
                                 line=dict(color=color, width=1.5, dash="dash")))
    fig.update_layout(title="Actual vs Predicted", height=320,
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"))
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    # Tactic coefficients bar
    with col_left:
        tactic_feats = [f for f in result["coef_df"]["feature"]
                        if f.endswith("_final") and result["coef_df"].set_index("feature").loc[f, "coefficient"] > 0]
        if tactic_feats:
            coef_sub = result["coef_df"][result["coef_df"]["feature"].isin(tactic_feats)].copy()
            coef_sub["label"] = coef_sub["feature"].str.replace("_final", "", regex=False)
            fig2 = px.bar(
                coef_sub.sort_values("coefficient"),
                x="coefficient", y="label", orientation="h",
                title="NNLS Tactic Coefficients (≥ 0 enforced)",
                color_discrete_sequence=["#378ADD"],
            )
            fig2.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    # Contribution decomp (test weeks)
    with col_right:
        contrib = result["contrib"].copy()
        tactic_contrib_cols = [c for c in contrib.columns
                                if c.endswith("_final") and c in contrib.columns]
        seasonal_cols = [c for c in contrib.columns if c.startswith(("sin_", "cos_"))]
        base_cols     = ["const"] + seasonal_cols

        if tactic_contrib_cols:
            contrib["Seasonality / Base"] = contrib[base_cols].sum(axis=1)
            contrib["Marketing"]          = contrib[tactic_contrib_cols].sum(axis=1)
            plot_cols = ["period_label", "Seasonality / Base", "Marketing"]
            melt = contrib[plot_cols].melt(id_vars="period_label",
                                           var_name="Component", value_name="Apps")
            fig3 = px.bar(melt, x="period_label", y="Apps", color="Component",
                          barmode="stack", title="Contribution decomp — test weeks",
                          color_discrete_sequence=["#B5D4F4", "#185FA5"])
            fig3.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                               legend=dict(orientation="h", y=-0.35, x=0.5, xanchor="center"))
            st.plotly_chart(fig3, use_container_width=True)

    with st.expander("📋 Contribution decomposition detail"):
        st.dataframe(contrib, use_container_width=True, hide_index=True)

    with st.expander("📋 Coefficient table"):
        st.dataframe(result["coef_df"], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Existing tab renderers (unchanged)
# ---------------------------------------------------------------------------

def render_channel_chart_expander(channel, chart_data, metric_label):
    st.markdown(f"#### {channel}")
    if chart_data.empty:
        st.info(f"No {channel.lower()} {metric_label.lower()} data available for the current filter.")
        return
    df_plot  = chart_data.reset_index()
    period_col = df_plot.columns[0]
    value_cols = [c for c in df_plot.columns if c != period_col]
    fig = px.line(df_plot, x=period_col, y=value_cols, markers=True, color_discrete_sequence=PALETTE)
    fig = apply_bottom_legend(fig)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 View data"):
        st.dataframe(df_plot, use_container_width=True, hide_index=True)

def render_product_bar_chart(channel, chart_data):
    st.markdown(f"#### {channel}")
    if chart_data.empty:
        st.info(f"No {channel.lower()} product spend data available.")
        return
    df_plot  = chart_data.reset_index()
    period_col = df_plot.columns[0]
    value_cols = [c for c in df_plot.columns if c != period_col]
    fig = px.bar(df_plot, x=period_col, y=value_cols, barmode="stack",
                 color_discrete_sequence=PALETTE,
                 labels={"value": "Spend ($)", "variable": "Tactic"})
    fig = apply_bottom_legend(fig)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 View data"):
        st.dataframe(df_plot, use_container_width=True, hide_index=True)

def render_model_panel(channel, result):
    st.markdown(f"#### {channel} model")
    mape_display = "N/A" if pd.isna(result.mape) else f"{format_metric(result.mape, 2)}%"
    st.metric("MAPE", mape_display)
    st.caption(
        f"**Y-axis:** `Applications_{channel}`  ·  "
        f"**X-axis:** channel spend tactics + time dummies "
        f"({', '.join(result.periods_used[:4])}{' ...' if len(result.periods_used) > 4 else ''})."
    )
    chart_frame = result.fitted_frame.set_index("period_label")[
        ["APPLICATIONS", "Predicted_Applications"]
    ]
    fig = px.line(chart_frame.reset_index(), x="period_label",
                  y=["APPLICATIONS", "Predicted_Applications"],
                  markers=True, color_discrete_sequence=PALETTE,
                  labels={"value": "Applications", "variable": "Series", "period_label": "Period"})
    fig = apply_bottom_legend(fig)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 Actual vs Predicted data"):
        st.dataframe(
            result.fitted_frame.rename(columns={
                "APPLICATIONS": "Actual_Applications",
                "Predicted_Applications": "Predicted_Applications",
                "TOTAL_SPEND": "Total_Spend",
            }),
            use_container_width=True, hide_index=True,
        )
    tactic_coefs = result.coefficients.loc[
        result.coefficients["feature"].isin(TACTIC_COLUMNS),
        ["feature", "coefficient", "p_value"],
    ].copy()
    if not tactic_coefs.empty:
        st.markdown("**Tactic Coefficients**")
        coef_fig = px.bar(
            tactic_coefs.sort_values("coefficient"),
            x="coefficient", y="feature", orientation="h",
            color="coefficient", color_continuous_scale="RdBu",
            title="Spend Tactic Coefficients",
            labels={"feature": "Tactic", "coefficient": "Coefficient"},
        )
        coef_fig = apply_bottom_legend(coef_fig)
        st.plotly_chart(coef_fig, use_container_width=True)
        with st.expander("📋 Coefficient table"):
            st.dataframe(tactic_coefs, use_container_width=True, hide_index=True)
    if result.narrative:
        st.markdown("**Interpretation**")
        for sentence in result.narrative:
            st.write(sentence)


# ---------------------------------------------------------------------------
# Tab 1 — Marketing Spend EDA
# ---------------------------------------------------------------------------

def render_tab_marketing_spend():
    st.header("Exploratory Data Analysis — Marketing Spend")
    try:
        df = auto_load_marketing_spend()
    except Exception as exc:
        st.error(f"Unable to load marketing spend data. Error: {exc}")
        return
    if df.attrs.get("source_label") == "fallback_modeling_workbook":
        st.info("Using fallback data derived from the consolidated modeling workbook.")
    st.caption(
        f"**{len(df):,}** rows · "
        f"**{df['DETAIL_TACTIC'].nunique()}** tactics · "
        f"**{df['STATE_CD'].nunique() if 'STATE_CD' in df.columns else '—'}** states"
    )
    st.subheader("1 · Spend by Tactic")
    tactic_df = ms_spend_by_tactic(df)
    chart_with_table(bar_h(tactic_df, x="Total Spend ($)", y="Tactic", title="Total Spend by Tactic"), tactic_df, "Spend by Tactic")
    st.subheader("2 · Spend by Channel")
    channel_df = ms_spend_by_channel(df)
    chart_with_table(bar_v(channel_df, x="Channel", y="Total Spend ($)", title="Total Spend by Channel"), channel_df, "Spend by Channel")
    st.subheader("3 · Spend by State")
    state_df = ms_spend_by_state(df)
    chart_with_table(bar_h(state_df, x="Total Spend ($)", y="State", title="Total Spend by State"), state_df, "Spend by State")
    st.subheader("4 · Spend by Product")
    prod_df = ms_spend_by_product(df)
    chart_with_table(bar_h(prod_df, x="Total Spend ($)", y="Product", title="Total Spend by Product"), prod_df, "Spend by Product")
    st.subheader("5 · Weekly Spend Over Time")
    time_df = ms_spend_over_time(df)
    if not time_df.empty:
        chart_with_table(line_chart(time_df, x="period", y="TOTAL_COST", title="Weekly Total Spend Over Time"),
                         time_df[["period", "TOTAL_COST"]].rename(columns={"TOTAL_COST": "Spend ($)"}), "Weekly Spend")
    st.subheader("6 · Tactic × Channel Spend Heatmap")
    matrix_df = ms_tactic_channel_matrix(df)
    if not matrix_df.empty:
        chart_with_table(heatmap_fig(matrix_df, "Spend Heatmap — Tactic × Channel"), matrix_df, "Tactic × Channel Matrix")
    st.subheader("7 · Spend by State & Tactic")
    st_tactic = ms_spend_by_state_tactic(df)
    if not st_tactic.empty:
        fig = px.bar(st_tactic, x="STATE_CD", y="TOTAL_COST", color="DETAIL_TACTIC",
                     title="Spend by State and Tactic", barmode="stack",
                     color_discrete_sequence=PALETTE, text_auto=".3s",
                     labels={"STATE_CD": "State", "TOTAL_COST": "Spend ($)", "DETAIL_TACTIC": "Tactic"})
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        chart_with_table(fig, st_tactic.rename(columns={"STATE_CD": "State", "DETAIL_TACTIC": "Tactic", "TOTAL_COST": "Spend ($)"}), "State × Tactic Spend")


# ---------------------------------------------------------------------------
# Tab 2 — Originations EDA
# ---------------------------------------------------------------------------

def render_tab_originations():
    st.header("Exploratory Data Analysis — Originations")
    try:
        df = auto_load_originations()
    except Exception as exc:
        st.error(f"Unable to load originations data. Error: {exc}")
        return
    if df.attrs.get("source_label") == "fallback_modeling_workbook":
        st.info("Using fallback outcomes derived from the consolidated modeling workbook.")
    total_apps = int(df["APPLICATIONS"].sum()) if "APPLICATIONS" in df.columns else 0
    total_appr = int(df["APPROVED"].sum())      if "APPROVED"     in df.columns else 0
    total_orig = int(df["ORIGINATIONS"].sum())  if "ORIGINATIONS" in df.columns else 0
    st.caption(f"**{len(df):,}** rows · Applications: **{total_apps:,}** · Approved: **{total_appr:,}** · Funded: **{total_orig:,}**")
    st.subheader("1 · Applications, Approvals & Funding by State")
    state_df    = od_metrics_by_state(df)
    metric_cols = [c for c in ["APPLICATIONS","APPROVED","ORIGINATIONS"] if c in state_df.columns]
    chart_with_table(bar_v(state_df, x="State", y=metric_cols, title="Applications / Approvals / Funding by State", barmode="group"), state_df, "Metrics by State")
    st.subheader("2 · Metrics by Channel")
    channel_df   = od_metrics_by_channel(df)
    metric_cols2 = [c for c in ["APPLICATIONS","APPROVED","ORIGINATIONS"] if c in channel_df.columns]
    chart_with_table(bar_v(channel_df, x="Channel", y=metric_cols2, title="Metrics by Channel", barmode="group"), channel_df, "Metrics by Channel")
    st.subheader("3 · Metrics by Product")
    prod_df      = od_metrics_by_product(df)
    metric_cols3 = [c for c in ["APPLICATIONS","APPROVED","ORIGINATIONS"] if c in prod_df.columns]
    melted = prod_df.melt(id_vars=["Product"], value_vars=metric_cols3, var_name="Metric", value_name="Count")
    chart_with_table(bar_h(melted, x="Count", y="Product", title="Metrics by Product", color="Metric"), prod_df, "Metrics by Product")
    st.subheader("4 · Weekly Trends")
    time_df      = od_metrics_over_time(df)
    if not time_df.empty:
        metric_cols4 = [c for c in ["APPLICATIONS","APPROVED","ORIGINATIONS"] if c in time_df.columns]
        chart_with_table(line_chart(time_df, x="period", y=metric_cols4, title="Weekly Applications, Approvals & Funding Over Time"), time_df[["period"]+metric_cols4], "Weekly Trends")
    st.subheader("5 · Approval & Funding Rate by State")
    funnel_df  = od_funnel_by_state(df)
    rate_cols  = [c for c in ["Approval Rate (%)","Funding Rate (%)"] if c in funnel_df.columns]
    if rate_cols:
        chart_with_table(bar_v(funnel_df, x="State", y=rate_cols, title="Approval Rate & Funding Rate by State (%)", barmode="group"), funnel_df[["State"]+rate_cols], "Funnel Rates by State")
    st.subheader("6 · Channel × State Breakdown")
    cs_df = od_channel_state_matrix(df)
    if not cs_df.empty and "APPLICATIONS" in cs_df.columns:
        fig = px.bar(cs_df, x="STATE_CD", y="APPLICATIONS", color="CHANNEL_CD", barmode="group",
                     title="Applications by State and Channel", color_discrete_sequence=PALETTE,
                     labels={"STATE_CD":"State","APPLICATIONS":"Applications","CHANNEL_CD":"Channel"})
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
        chart_with_table(fig, cs_df.rename(columns={"STATE_CD":"State","CHANNEL_CD":"Channel"}), "Channel × State Data")


# ---------------------------------------------------------------------------
# Tab 3 — Marketing Analysis V1 (original OLS)
# ---------------------------------------------------------------------------

def render_tab_marketing_analysis():
    uploaded_file = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx"], key="model_upload")
    time_grain    = st.selectbox("Aggregation", TIME_GRAINS)

    if uploaded_file is not None:
        try:
            data = cached_load_data(uploaded_file.getvalue(), time_grain=time_grain)
        except Exception as exc:
            st.error(f"Unable to load workbook: {exc}")
            return
    elif DEFAULT_DATA_PATH.exists():
        try:
            data = auto_load_modeling(str(DEFAULT_DATA_PATH), time_grain=time_grain)
        except Exception as exc:
            st.error(f"Unable to load default workbook: {exc}")
            return
    else:
        st.info("Upload the consolidated workbook (.xlsx) to begin analysis.")
        return

    state_options = sorted(data["STATE_CD"].dropna().unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        state   = st.selectbox("STATE",             state_options)
    with c2:
        product = st.selectbox("PRODUCT (optional)", get_available_products(data, state))

    st.subheader("Section 1: Marketing Spend Consistency")
    tactic_series = build_tactic_time_series(data, time_grain=time_grain, state=state,
                                              product=None if product == PRODUCT_ALL_LABEL else product)
    l, r = st.columns(2)
    with l: render_channel_chart_expander("DIGITAL",  tactic_series["DIGITAL"],  "spend")
    with r: render_channel_chart_expander("PHYSICAL", tactic_series["PHYSICAL"], "spend")

    st.subheader("Section 2: Product-Level Marketing Consistency")
    if product == PRODUCT_ALL_LABEL:
        st.info("Select a specific product to compare DIGITAL and PHYSICAL spend over time.")
    else:
        product_series = build_product_spend_series(data, time_grain=time_grain, state=state, product=product)
        l, r = st.columns(2)
        with l: render_product_bar_chart("DIGITAL",  product_series["DIGITAL"])
        with r: render_product_bar_chart("PHYSICAL", product_series["PHYSICAL"])
        product_comparison = build_channel_comparison_table(data, time_grain=time_grain, state=state, product=product)
        with st.expander("📋 DIGITAL vs PHYSICAL comparison"):
            st.dataframe(product_comparison, use_container_width=True, hide_index=True)

    st.subheader("Section 3: Applications, Approvals & Funding Consistency")
    application_series = build_application_series(data, time_grain=time_grain, state=state,
                                                   product=None if product == PRODUCT_ALL_LABEL else product)
    l, r = st.columns(2)
    with l: render_channel_chart_expander("DIGITAL",  application_series["DIGITAL"],  "outcomes")
    with r: render_channel_chart_expander("PHYSICAL", application_series["PHYSICAL"], "outcomes")

    st.subheader("Modeling Section — V1 (OLS)")
    model_product = st.selectbox("Product (modeling)", get_available_products(data, state),
                                 key="model_product",
                                 help="Each channel modeled separately with OLS.")
    st.caption(
        "**Model spec — Y:** Applications by Channel  ·  "
        "**X:** spend tactics + week/bi-weekly time dummies. "
        "F01/W_1 dropped as baseline."
    )
    modeling_df = prepare_modeling_dataset(data, time_grain=time_grain, state=state,
                                           product=None if model_product == PRODUCT_ALL_LABEL else model_product)
    if modeling_df.empty:
        st.warning("No rows available for the current modeling filter.")
        return
    try:
        model_results = fit_channel_models(modeling_df)
    except Exception as exc:
        st.error(f"Model fitting failed: {exc}")
        return

    l, r = st.columns(2)
    for col, channel in zip((l, r), CHANNELS):
        with col:
            if channel not in model_results:
                st.info(f"No {channel.lower()} observations available.")
            else:
                render_model_panel(channel, model_results[channel])


# ---------------------------------------------------------------------------
# Tab 4 — Marketing Analysis V2 (NNLS MMM — all 11 steps)
# ---------------------------------------------------------------------------

def render_tab_mmm_v2():
    st.header("Marketing Analysis V2 — Advanced MMM (11-step pipeline)")

    with st.expander("ℹ️ How V2 differs from V1", expanded=False):
        st.markdown("""
| | V1 (OLS) | V2 (MMM — this tab) |
|---|---|---|
| Seasonality | 52 week dummies | 4 Fourier features (sin/cos) |
| Prescreen | Raw weekly spend | Circular spread across 5 weeks |
| Digital tactics | Raw spend | Adstock (0.5 decay) + Hill saturation |
| Optimizer | OLS (allows negatives) | NNLS (all coefficients ≥ 0) |
| Diagnostics | MAPE only | R², Adj-R², MAE%, Durbin-Watson |
| Output | Coefficients | Coefficients + contribution decomp |
        """)

    # Controls
    uploaded_file = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx"], key="v2_upload")
    c1, c2, c3 = st.columns([2, 2, 3])

    if uploaded_file is not None:
        try:
            data = cached_load_data(uploaded_file.getvalue(), time_grain="Weekly")
        except Exception as exc:
            st.error(f"Unable to load workbook: {exc}")
            return
    elif DEFAULT_DATA_PATH.exists():
        try:
            data = auto_load_modeling(str(DEFAULT_DATA_PATH), time_grain="Weekly")
        except Exception as exc:
            st.error(f"Unable to load default workbook: {exc}")
            return
    else:
        st.info("Upload the consolidated workbook (.xlsx) to begin analysis.")
        return

    state_options = sorted(data["STATE_CD"].dropna().unique().tolist())
    with c1:
        state   = st.selectbox("State",   state_options,                        key="v2_state")
    with c2:
        product = st.selectbox("Product", get_available_products(data, state),  key="v2_product")
    with c3:
        scheme_label = st.selectbox(
            "Prescreen spread weights  (Step 4)",
            list(WEIGHT_SCHEMES.keys()),
            index=0,
            key="v2_weights",
            help="Controls how monthly Prescreen spend is distributed across 5 weeks. lag2_peak is data-confirmed best.",
        )

    weights = WEIGHT_SCHEMES[scheme_label]

    # Show weight visualisation
    with st.expander("📊 Weight distribution preview", expanded=False):
        w_df = pd.DataFrame({
            "Lag":   [f"Week 0 (drop week)", "Week +1", "Week +2", "Week +3", "Week +4"],
            "Weight": weights,
            "Share":  [f"{w*100:.0f}%" for w in weights],
        })
        fig_w = px.bar(w_df, x="Lag", y="Weight", text="Share",
                       title=f"Prescreen spread — {scheme_label.split('[')[0].strip()}",
                       color_discrete_sequence=["#185FA5"])
        fig_w.update_layout(height=250, margin=dict(l=10,r=10,t=40,b=10), yaxis_tickformat=".0%")
        st.plotly_chart(fig_w, use_container_width=True)

    # Filter data
    filtered = data[data["STATE_CD"] == state].copy()
    if product != PRODUCT_ALL_LABEL:
        filtered = filtered[filtered["PRODUCT_CD"] == product].copy()

    if filtered.empty:
        st.warning("No data for the selected state/product.")
        return

    # Pipeline info banner
    st.info(
        f"**Pipeline:** Train on {TRAIN_YEARS_V2} → Test on remaining weeks  ·  "
        f"Adstock decay = {ADSTOCK_DECAY}  ·  Fourier k={FOURIER_K}  ·  "
        f"Weights = {weights}"
    )

    # Run per channel
    for channel in CHANNELS:
        st.subheader(f"{channel} channel")
        with st.spinner(f"Running V2 MMM for {channel}…"):
            result = run_mmm_for_channel(filtered, channel, weights)

        if result is None:
            st.info(f"Insufficient data for {channel} modeling (need ≥ 5 train weeks + test weeks).")
            continue

        avg_apps = result["avg_test_apps"]
        render_mmm_channel_result(result, avg_apps)
        st.divider()


def render_tab_mmm_v3():
    st.header("Marketing Analysis V3 — All States, No Fourier")

    with st.expander("ℹ️ How V3 differs from V2", expanded=False):
        st.markdown("""
| | V2 | V3 |
|---|---|---|
| State coverage | 4-state workbook only | all usable states from raw data |
| Train period | 2024 + 2025 | 2024 + 2025 |
| Test period | all non-train years | rolling 2026 periods |
| Seasonality control | Fourier pairs | week/fortnight dummies |
| Prescreen handling | circular spread | circular spread |
| Digital tactics | adstock + Hill | adstock + Hill |
| Optimizer | NNLS | NNLS |
        """)

    time_grain = st.selectbox("Time grain", TIME_GRAINS, key="v3_time_grain")
    time_settings = get_v3_time_settings(time_grain)

    try:
        base = auto_prepare_v3_dataset(time_grain=time_grain)
    except Exception as exc:
        st.error(f"Unable to prepare all-state MMM dataset from raw files: {exc}")
        return

    state_summary = (
        base.assign(
            is_train=base["ISO_YEAR"].isin(TRAIN_YEARS_V3).astype(int),
            is_test=base["ISO_YEAR"].isin(TEST_YEARS_V3).astype(int),
        )
        .groupby("STATE_CD", dropna=False)
        .agg(
            spend_total=("TOTAL_SPEND", "sum"),
            applications_total=("APPLICATIONS", "sum"),
            train_rows=("is_train", "sum"),
            test_rows=("is_test", "sum"),
        )
        .reset_index()
    )
    eligible_states = (
        state_summary.loc[
            (state_summary["spend_total"] > 0)
            & (state_summary["applications_total"] > 0)
            & (state_summary["train_rows"] > 0)
            & (state_summary["test_rows"] > 0),
            "STATE_CD",
        ]
        .sort_values()
        .tolist()
    )

    if not eligible_states:
        st.warning(f"No states have enough raw {time_grain.lower()} data for the V3 train/test design.")
        return

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        state = st.selectbox("State", eligible_states, key="v3_state")
    with c2:
        product = st.selectbox("Product", get_available_products(base, state), key="v3_product")
    with c3:
        scheme_label = st.selectbox(
            "Prescreen spread weights",
            list(WEIGHT_SCHEMES.keys()),
            index=0,
            key="v3_weights",
            help="Controls how Prescreen spend is distributed across 5 weeks.",
        )

    weights = WEIGHT_SCHEMES[scheme_label]
    filtered = base[base["STATE_CD"] == state].copy()
    if product != PRODUCT_ALL_LABEL:
        filtered = filtered[filtered["PRODUCT_CD"] == product].copy()

    if filtered.empty:
        st.warning("No raw weekly data for the selected state/product.")
        return

    state_detail = state_summary.loc[state_summary["STATE_CD"] == state].iloc[0]
    st.info(
        f"**Pipeline:** train on {TRAIN_YEARS_V3} · test on {TEST_YEARS_V3} · "
        f"seasonality via {time_settings['seasonality_label']} with no Fourier features · "
        f"rolling 2026 {time_settings['test_unit_label']} (each period learns from all prior 2026 periods) · "
        f"adstock decay = {ADSTOCK_DECAY} · usable states = {len(eligible_states)} · "
        f"selected state train rows = {int(state_detail['train_rows'])}, test rows = {int(state_detail['test_rows'])}"
    )

    for channel in CHANNELS:
        st.subheader(f"{channel} channel")
        with st.spinner(f"Running V3 MMM for {channel}…"):
            result = run_mmm_rolling_v3(
                filtered,
                channel,
                weights,
                time_grain=time_grain,
                train_years=TRAIN_YEARS_V3,
                test_years=TEST_YEARS_V3,
            )

        if result is None:
            st.info(f"Insufficient data for {channel} modeling under the V3 split.")
            continue

        avg_apps = result["avg_test_apps"]
        render_mmm_rolling_result(result, avg_apps, key_prefix=f"v3-{time_grain.lower()}-{channel.lower()}")
        st.divider()


def render_tab_mmm_v4():
    st.header("Marketing Analysis V4 — Rolled-Up Products, Direct Mail Separate")

    with st.expander("ℹ️ How V4 differs from V3", expanded=False):
        st.markdown("""
| | V3 | V4 |
|---|---|---|
| Product scope | all products in raw data | rolled-up products only |
| Direct Mail | not modeled separately | excluded from MMM and tracked via `DM Data.csv` |
| Train period | 2024 + 2025 | 2024 + 2025 |
| Test period | rolling 2026 periods | rolling 2026 periods |
| Seasonality control | week/fortnight dummies | week/fortnight dummies |
| Optimizer | NNLS | NNLS |
        """)

    time_grain = st.selectbox("Time grain", TIME_GRAINS, key="v4_time_grain")
    time_settings = get_v3_time_settings(time_grain)

    try:
        base = auto_prepare_v4_dataset(time_grain=time_grain)
    except Exception as exc:
        st.error(f"Unable to prepare V4 dataset: {exc}")
        return

    try:
        dm_df = auto_load_dm_data()
    except Exception as exc:
        dm_df = pd.DataFrame()
        st.warning(f"Direct Mail file could not be loaded: {exc}")

    state_summary = (
        base.assign(
            is_train=base["ISO_YEAR"].isin(TRAIN_YEARS_V3).astype(int),
            is_test=base["ISO_YEAR"].isin(TEST_YEARS_V3).astype(int),
        )
        .groupby("STATE_CD", dropna=False)
        .agg(
            spend_total=("TOTAL_SPEND", "sum"),
            applications_total=("APPLICATIONS", "sum"),
            train_rows=("is_train", "sum"),
            test_rows=("is_test", "sum"),
        )
        .reset_index()
    )
    eligible_states = (
        state_summary.loc[
            (state_summary["spend_total"] > 0)
            & (state_summary["applications_total"] > 0)
            & (state_summary["train_rows"] > 0)
            & (state_summary["test_rows"] > 0),
            "STATE_CD",
        ]
        .sort_values()
        .tolist()
    )

    if not eligible_states:
        st.warning(f"No states have enough raw {time_grain.lower()} rolled-up data for V4.")
        return

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        state = st.selectbox("State", eligible_states, key="v4_state")
    with c2:
        product = st.selectbox("Rolled-up product", get_available_rolled_up_products(base, state), key="v4_product")
    with c3:
        scheme_label = st.selectbox(
            "Prescreen spread weights",
            list(WEIGHT_SCHEMES.keys()),
            index=0,
            key="v4_weights",
            help="Controls how Prescreen spend is distributed across 5 weeks.",
        )

    weights = WEIGHT_SCHEMES[scheme_label]
    filtered = base[base["STATE_CD"] == state].copy()
    if product != PRODUCT_ALL_LABEL:
        filtered = filtered[filtered["PRODUCT_CD"] == product].copy()

    if filtered.empty:
        st.warning("No rolled-up product data for the selected V4 state/product.")
        return

    state_detail = state_summary.loc[state_summary["STATE_CD"] == state].iloc[0]
    product_mapping = expand_rollup_product(product) if product != PRODUCT_ALL_LABEL else []
    mapping_label = ", ".join(product_mapping) if product_mapping else "all available DM offer products"
    st.info(
        f"**Pipeline:** train on {TRAIN_YEARS_V3} · test on {TEST_YEARS_V3} · "
        f"seasonality via {time_settings['seasonality_label']} · "
        f"rolling 2026 {time_settings['test_unit_label']} · "
        f"rolled-up products only · Direct Mail excluded from MMM and loaded separately from `{DM_DATA_PATH.name}` · "
        f"selected state train rows = {int(state_detail['train_rows'])}, test rows = {int(state_detail['test_rows'])} · "
        f"DM product mapping = {mapping_label}"
    )

    if not dm_df.empty:
        dm_summary = summarize_dm_data(dm_df, state=state, product=product, time_grain=time_grain)
        with st.expander("📬 Direct Mail summary (separate input)"):
            if dm_summary.empty:
                st.info("No Direct Mail rows matched the current state/product selection.")
            else:
                dm_metrics = ["UNIQUE_RESERVATION_CODE_COUNT", "UNIQUE_APPLICANTS", "UNIQUE_APPROVED", "UNIQUE_FUNDED"]
                dm_totals = dm_summary[dm_metrics].sum()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("DM Mailers", f"{int(dm_totals['UNIQUE_RESERVATION_CODE_COUNT']):,}")
                m2.metric("DM Applicants", f"{int(dm_totals['UNIQUE_APPLICANTS']):,}")
                m3.metric("DM Approved", f"{int(dm_totals['UNIQUE_APPROVED']):,}")
                m4.metric("DM Funded", f"{int(dm_totals['UNIQUE_FUNDED']):,}")
                dm_plot = dm_summary.melt(
                    id_vars=["period_label", "DM_CHANNEL_CD"],
                    value_vars=["UNIQUE_APPLICANTS", "UNIQUE_APPROVED", "UNIQUE_FUNDED"],
                    var_name="Metric",
                    value_name="Count",
                )
                fig_dm = px.line(
                    dm_plot,
                    x="period_label",
                    y="Count",
                    color="Metric",
                    line_dash="DM_CHANNEL_CD",
                    markers=True,
                    title="Direct Mail outcomes over time",
                    color_discrete_sequence=PALETTE,
                )
                fig_dm.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_dm, use_container_width=True)
                st.dataframe(dm_summary, use_container_width=True, hide_index=True)

    for channel in CHANNELS:
        st.subheader(f"{channel} channel")
        with st.spinner(f"Running V4 MMM for {channel}…"):
            result = run_mmm_rolling_v3(
                filtered,
                channel,
                weights,
                time_grain=time_grain,
                train_years=TRAIN_YEARS_V3,
                test_years=TEST_YEARS_V3,
            )

        if result is None:
            st.info(f"Insufficient data for {channel} modeling under the V4 split.")
            continue

        avg_apps = result["avg_test_apps"]
        render_mmm_rolling_result(result, avg_apps, key_prefix=f"v4-{time_grain.lower()}-{channel.lower()}")
        st.divider()


# ---------------------------------------------------------------------------
# Main — render all tabs
# ---------------------------------------------------------------------------

st.title("Marketing Analytics and Modeling")
st.caption("Channel-specific analysis and regression for DIGITAL and PHYSICAL applications.")

tabs = st.tabs([
    "📊 Marketing Spend EDA",
    "📈 Originations EDA",
    "🔬 Marketing Analysis",
    "🧪 Marketing Analysis V2",
    "🧩 Marketing Analysis V3",
    "🧱 Marketing Analysis V4",
])

with tabs[0]:
    render_tab_marketing_spend()

with tabs[1]:
    render_tab_originations()

with tabs[2]:
    render_tab_marketing_analysis()

with tabs[3]:
    render_tab_mmm_v2()

with tabs[4]:
    render_tab_mmm_v3()

with tabs[5]:
    render_tab_mmm_v4()
