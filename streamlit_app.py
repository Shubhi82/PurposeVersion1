from __future__ import annotations

import inspect
from pathlib import Path

import data_processing as dp
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

build_application_series = dp.build_application_series
build_channel_comparison_table = dp.build_channel_comparison_table
build_modeling_frame = dp.build_modeling_frame
build_v7_modeling_frame = getattr(dp, "build_v7_modeling_frame", dp.build_modeling_frame)
build_product_spend_series = dp.build_product_spend_series
build_tactic_time_series = dp.build_tactic_time_series
exclude_direct_mail_rows = dp.exclude_direct_mail_rows
filter_rolled_up_products = dp.filter_rolled_up_products
load_marketing_spend_data = dp.load_marketing_spend_data
load_marketing_spend_raw = dp.load_marketing_spend_raw
load_dm_data = dp.load_dm_data
load_modeling_data = dp.load_modeling_data
load_originations_data = dp.load_originations_data
load_originations_raw = dp.load_originations_raw
ms_spend_by_channel = dp.ms_spend_by_channel
ms_spend_by_product = dp.ms_spend_by_product
ms_spend_by_state = dp.ms_spend_by_state
ms_spend_by_state_tactic = dp.ms_spend_by_state_tactic
ms_spend_by_tactic = dp.ms_spend_by_tactic
ms_spend_over_time = dp.ms_spend_over_time
ms_tactic_channel_matrix = dp.ms_tactic_channel_matrix
od_channel_state_matrix = dp.od_channel_state_matrix
od_funnel_by_state = dp.od_funnel_by_state
od_metrics_by_channel = dp.od_metrics_by_channel
od_metrics_by_product = dp.od_metrics_by_product
od_metrics_by_state = dp.od_metrics_by_state
od_metrics_over_time = dp.od_metrics_over_time
prepare_modeling_dataset = dp.prepare_modeling_dataset
prepare_raw_mmm_dataset = getattr(dp, "prepare_raw_mmm_dataset", None)
run_all_configs_for_entity = getattr(dp, "run_all_configs_for_entity", None)
run_ols_configs_for_entity = getattr(dp, "run_ols_configs_for_entity", None)
run_v6_iterations_for_entity = getattr(dp, "run_v6_iterations_for_entity", None)
V6_ITERATIONS = getattr(dp, "V6_ITERATIONS", [])
summarize_dm_data = dp.summarize_dm_data
from modeling import fit_channel_models
from utils import (
    CHANNELS,
    DEFAULT_DATA_PATH,
    DIAGNOSTICS_DIGITAL_PATH,
    DIAGNOSTICS_PHYSICAL_PATH,
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


def render_version_intro(title: str, steps: list[str], note: str | None = None) -> None:
    st.header(title)
    if note:
        st.caption(note)
    st.markdown("**What This Version Does**")
    for idx, step in enumerate(steps, start=1):
        st.write(f"{idx}. {step}")


def _call_fit_v6_compat(
    fit_func,
    entity_df: pd.DataFrame,
    channel: str,
    cfg: dict,
    extra_kwargs: dict | None = None,
):
    """
    Call fit_v6_iteration defensively.
    Streamlit Cloud can briefly load mismatched app/module versions during deploys,
    so only pass kwargs supported by the currently loaded function signature.
    """
    kwargs = {
        "dummy_family": cfg["dummy_family"],
        "prescreen_transform": cfg["prescreen_transform"],
        "add_interaction": cfg["add_interaction"],
        "drop_prescreen": cfg["drop_prescreen"],
        "log_tactics": cfg.get("log_tactics"),
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    try:
        signature = inspect.signature(fit_func)
        supported = {name for name in signature.parameters}
        kwargs = {key: value for key, value in kwargs.items() if key in supported}
    except (TypeError, ValueError):
        # If the signature cannot be inspected, fall back to the full kwargs set.
        pass
    return fit_func(entity_df, channel, **kwargs)

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
    try:
        return load_modeling_data(Path(path), time_grain=time_grain)
    except TypeError as exc:
        if "unexpected keyword argument 'time_grain'" not in str(exc):
            raise
        return load_modeling_data(Path(path))

@st.cache_data(show_spinner="Building modeling frame from raw files…")
def cached_build_modeling_frame(channel: str, product: str = "") -> pd.DataFrame:
    return build_modeling_frame(channel, product or None)


@st.cache_data(show_spinner="Building V7 NON-DM modeling frame…")
def cached_build_v7_modeling_frame(channel: str) -> pd.DataFrame:
    return build_v7_modeling_frame(channel)


@st.cache_data(show_spinner=False)
def cached_load_data(source: bytes, time_grain: str = "Weekly"):
    try:
        return load_modeling_data(source, time_grain=time_grain)
    except TypeError as exc:
        if "unexpected keyword argument 'time_grain'" not in str(exc):
            raise
        return load_modeling_data(source)

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


def _prepare_workbook_period_columns(df: pd.DataFrame, time_grain: str) -> pd.DataFrame:
    prepared = df.copy()
    prepared["ISO_YEAR"] = pd.to_numeric(prepared["ISO_YEAR"], errors="coerce").fillna(0).astype(int)
    prepared["ISO_WEEK"] = pd.to_numeric(prepared["ISO_WEEK"], errors="coerce").fillna(0).astype(int)

    if time_grain in {"Fortnight", "Fortnightly"}:
        prepared["period_value"] = np.minimum(((prepared["ISO_WEEK"] - 1) // 2) + 1, 26).astype(int)
        prepared["period_label"] = (
            prepared["ISO_YEAR"].astype(str) + "-BW" + prepared["period_value"].astype(str).str.zfill(2)
        )
    else:
        prepared["period_value"] = prepared["ISO_WEEK"].astype(int)
        prepared["period_label"] = (
            prepared["ISO_YEAR"].astype(str) + "-W" + prepared["period_value"].astype(str).str.zfill(2)
    )
    return prepared


def aggregate_v4_filtered_data(df: pd.DataFrame, time_grain: str) -> pd.DataFrame:
    """After V4 rolled-up filtering, collapse to year-period-channel before modeling."""
    prepared = _prepare_workbook_period_columns(df, time_grain)
    settings = get_v3_time_settings(time_grain)
    seasonal_features = list(settings["seasonal_features"])
    group_cols = ["ISO_YEAR", "period_value", "CHANNEL_CD"]
    sum_cols = [col for col in [*TACTIC_COLUMNS, *OUTCOME_COLUMNS, "TOTAL_SPEND"] if col in prepared.columns]
    if not sum_cols:
        return pd.DataFrame()

    aggregated = (
        prepared.groupby(group_cols, dropna=False)[sum_cols]
        .sum()
        .reset_index()
        .sort_values(["ISO_YEAR", "period_value", "CHANNEL_CD"])
        .reset_index(drop=True)
    )

    if "STATE_CD" in df.columns and not df["STATE_CD"].dropna().empty:
        aggregated["STATE_CD"] = str(df["STATE_CD"].dropna().iloc[0]).strip().upper()
    if "PRODUCT_CD" in df.columns and not df["PRODUCT_CD"].dropna().empty:
        products = sorted(df["PRODUCT_CD"].dropna().astype(str).str.strip().unique().tolist())
        aggregated["PRODUCT_CD"] = products[0] if len(products) == 1 else "Rolled Up"

    if time_grain in {"Fortnight", "Fortnightly"}:
        aggregated["ISO_WEEK"] = ((aggregated["period_value"] - 1) * 2) + 1
        aggregated["period_label"] = (
            aggregated["ISO_YEAR"].astype(str) + "-BW" + aggregated["period_value"].astype(str).str.zfill(2)
        )
    else:
        aggregated["ISO_WEEK"] = aggregated["period_value"].astype(int)
        aggregated["period_label"] = (
            aggregated["ISO_YEAR"].astype(str) + "-W" + aggregated["period_value"].astype(str).str.zfill(2)
        )

    for feature in seasonal_features:
        aggregated[feature] = 0.0
    for idx, row in aggregated.iterrows():
        dummy_name = seasonal_features[row["period_value"] - 2] if row["period_value"] >= 2 and row["period_value"] - 2 < len(seasonal_features) else None
        if dummy_name and dummy_name in aggregated.columns:
            aggregated.at[idx, dummy_name] = 1.0

    return aggregated


def run_workbook_rolling_regression(
    df: pd.DataFrame,
    channel: str,
    time_grain: str = "Weekly",
    train_years: list[int] | None = None,
    test_years: list[int] | None = None,
    min_train_rows: int = 5,
) -> dict | None:
    """Simple rolling regression using workbook tactics + workbook time dummies."""
    train_years = TRAIN_YEARS_V3 if train_years is None else train_years
    test_years = TEST_YEARS_V3 if test_years is None else test_years

    ch_df = df[df["CHANNEL_CD"] == channel].copy()
    if ch_df.empty:
        return None

    ch_df = _prepare_workbook_period_columns(ch_df, time_grain)
    ch_df = ch_df[ch_df["ISO_YEAR"].isin(train_years + test_years)].copy()
    ch_df = ch_df.sort_values(["ISO_YEAR", "period_value"]).reset_index(drop=True)

    train = ch_df[ch_df["ISO_YEAR"].isin(train_years)].copy()
    test = ch_df[ch_df["ISO_YEAR"].isin(test_years)].copy()
    if len(train) < min_train_rows or test.empty:
        return None

    seasonal_features = list(get_v3_time_settings(time_grain)["seasonal_features"])
    tactic_features = [
        col for col in TACTIC_COLUMNS
        if col in ch_df.columns and ch_df[col].fillna(0).abs().sum() > 0
    ]
    features = tactic_features + [col for col in seasonal_features if col in ch_df.columns]
    if not features:
        return None

    baseline_model = LinearRegression()
    baseline_model.fit(train[features].fillna(0.0).values, train["APPLICATIONS"].fillna(0.0).values)
    train["Predicted"] = np.clip(baseline_model.predict(train[features].fillna(0.0).values), 0, None)

    train_period_fit = _summarize_mmm_fit(train[["period_label", "APPLICATIONS", "Predicted"]], "Train (baseline)")
    coef_df = pd.DataFrame({
        "feature": ["Intercept", *features],
        "coefficient": [float(baseline_model.intercept_), *baseline_model.coef_.astype(float).tolist()],
    })

    test_keys = (
        test[["ISO_YEAR", "period_value", "period_label"]]
        .drop_duplicates()
        .sort_values(["ISO_YEAR", "period_value"])
        .reset_index(drop=True)
    )
    rolling_frames: list[pd.DataFrame] = []
    rolling_detail_rows: list[dict[str, object]] = []

    for key in test_keys.itertuples(index=False):
        year = int(key.ISO_YEAR)
        period_value = int(key.period_value)
        period_label = str(key.period_label)

        prior_test_mask = (
            (ch_df["ISO_YEAR"].isin(test_years))
            & (
                (ch_df["ISO_YEAR"] < year)
                | ((ch_df["ISO_YEAR"] == year) & (ch_df["period_value"] < period_value))
            )
        )
        step_train = ch_df[ch_df["ISO_YEAR"].isin(train_years) | prior_test_mask].copy()
        period_rows = ch_df[(ch_df["ISO_YEAR"] == year) & (ch_df["period_value"] == period_value)].copy()
        if len(step_train) < min_train_rows or period_rows.empty:
            continue

        step_model = LinearRegression()
        step_model.fit(step_train[features].fillna(0.0).values, step_train["APPLICATIONS"].fillna(0.0).values)
        period_rows["Predicted"] = np.clip(
            step_model.predict(period_rows[features].fillna(0.0).values),
            0,
            None,
        )
        rolling_frames.append(period_rows[["period_label", "APPLICATIONS", "Predicted"]].copy())

        rolling_detail_rows.append({
            "period_label": period_label,
            "Actual": float(period_rows["APPLICATIONS"].sum()),
            "Predicted": float(period_rows["Predicted"].sum()),
            "Train Rows": int(len(step_train)),
            "Predicted Rows": int(len(period_rows)),
            "Solver": "LinearRegression (OLS)",
        })

    if not rolling_frames:
        return None

    rolling_raw = pd.concat(rolling_frames, ignore_index=True)
    rolling_fit = _summarize_mmm_fit(rolling_raw, "Test (rolling)")
    fitted = pd.concat([train_period_fit, rolling_fit], ignore_index=True)

    rolling_detail = pd.DataFrame(rolling_detail_rows)
    y_train = train["APPLICATIONS"].to_numpy(dtype=float)
    yhat_train = train["Predicted"].to_numpy(dtype=float)
    y_test = rolling_detail["Actual"].to_numpy(dtype=float)
    yhat_test = rolling_detail["Predicted"].to_numpy(dtype=float)

    n_train, p = len(y_train), len(features)
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
        "solver": "LinearRegression (OLS)",
        "avg_test_apps": float(np.mean(y_test)) if len(y_test) > 0 else 0,
    }


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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train R²", f"{result['train_r2']:.3f}")
    c2.metric("Rolling Test R²", f"{result['test_r2']:.3f}")
    c3.metric("Rolling Test MAPE", f"{result['test_mape']:.2f}%")
    c4.metric("Rolling Test MAE %", f"{mae_pct:.1f}%")

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
        coef_sub = result["coef_df"][result["coef_df"]["feature"].isin(TACTIC_COLUMNS)].copy()
        if coef_sub.empty:
            coef_sub = result["coef_df"][result["coef_df"]["feature"].str.endswith("_final")].copy()
        coef_sub = coef_sub.loc[coef_sub["coefficient"] != 0].copy()
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
    render_version_intro(
        "Version 1 — Base Regression from Modeling Workbook",
        [
            "Uses the default modeling workbook sheet for the selected Weekly or Fortnight view.",
            "Filters the selected state and optional product.",
            "Fits separate DIGITAL and PHYSICAL OLS models on the full selected history.",
        ],
        note="Simple baseline regression using spend tactics and time dummies.",
    )
    time_grain    = st.selectbox("Aggregation", TIME_GRAINS)

    if not DEFAULT_DATA_PATH.exists():
        st.error("Default modeling workbook was not found.")
        return
    try:
        data = auto_load_modeling(str(DEFAULT_DATA_PATH), time_grain=time_grain)
    except Exception as exc:
        st.error(f"Unable to load default workbook: {exc}")
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
    render_version_intro(
        "Version 2 — Advanced MMM from Modeling Workbook",
        [
            "Uses the default modeling workbook sheet for the selected Weekly or Fortnight view.",
            "Applies the advanced MMM logic with Fourier, Prescreen spread, adstock, and Hill saturation.",
            "Fits separate DIGITAL and PHYSICAL channel models and compares train vs test fit.",
        ],
        note="Advanced version for transformed marketing response modeling.",
    )

    time_grain = st.selectbox("Aggregation", TIME_GRAINS, key="v2_time_grain")
    c1, c2, c3 = st.columns([2, 2, 3])

    if not DEFAULT_DATA_PATH.exists():
        st.error("Default modeling workbook was not found.")
        return
    try:
        data = auto_load_modeling(str(DEFAULT_DATA_PATH), time_grain=time_grain)
    except Exception as exc:
        st.error(f"Unable to load default workbook: {exc}")
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
        f"**Pipeline:** workbook sheet = {time_grain} · Train on {TRAIN_YEARS_V2} → Test on remaining periods  ·  "
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
    render_version_intro(
        "Version 3 — Simple Rolling Regression from Modeling Workbook",
        [
            "Uses the default workbook sheet for the selected Weekly or Fortnight view.",
            "Trains on 2024 and 2025, then rolls forward through 2026 one period at a time.",
            "Uses simple regression with workbook tactic columns and workbook time dummies only.",
        ],
        note="No upload, no Fourier, no lag weights, and no Direct Mail removal in this version.",
    )

    time_grain = st.selectbox("Time grain", TIME_GRAINS, key="v3_time_grain")
    time_settings = get_v3_time_settings(time_grain)

    if not DEFAULT_DATA_PATH.exists():
        st.error("Default modeling workbook was not found.")
        return
    try:
        base = auto_load_modeling(str(DEFAULT_DATA_PATH), time_grain=time_grain)
    except Exception as exc:
        st.error(f"Unable to load default workbook: {exc}")
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
        st.warning(f"No states have enough {time_grain.lower()} workbook data for the V3 train/test design.")
        return

    c1, c2 = st.columns(2)
    with c1:
        state = st.selectbox("State", eligible_states, key="v3_state")
    with c2:
        product = st.selectbox("Product", get_available_products(base, state), key="v3_product")
    filtered = base[base["STATE_CD"] == state].copy()
    if product != PRODUCT_ALL_LABEL:
        filtered = filtered[filtered["PRODUCT_CD"] == product].copy()

    if filtered.empty:
        st.warning("No workbook data for the selected state/product.")
        return

    state_detail = state_summary.loc[state_summary["STATE_CD"] == state].iloc[0]
    st.info(
        f"**Pipeline:** train on {TRAIN_YEARS_V3} · test on {TEST_YEARS_V3} · "
        f"seasonality via {time_settings['seasonality_label']} from the workbook · "
        f"rolling 2026 {time_settings['test_unit_label']} (each period learns from all prior 2026 periods) · "
        f"simple regression only · usable states = {len(eligible_states)} · "
        f"selected state train rows = {int(state_detail['train_rows'])}, test rows = {int(state_detail['test_rows'])}"
    )

    for channel in CHANNELS:
        st.subheader(f"{channel} channel")
        with st.spinner(f"Running V3 regression for {channel}…"):
            result = run_workbook_rolling_regression(
                filtered,
                channel,
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
    render_version_intro(
        "Version 4 — Rolled-Up Product Regression from Modeling Workbook",
        [
            "Uses the default workbook sheet for the selected Weekly or Fortnight view.",
            "Keeps only rolled-up products like ILP/FLC, ILP/PDL, LOC/FLC, and PDL/FLC.",
            "Re-aggregates the filtered rolled-up data to year-period-channel before modeling.",
            "Runs rolling 2026 regression separately for DIGITAL and PHYSICAL and requires at least 20 train rows.",
        ],
        note="No upload and no Direct Mail summary section in this version.",
    )

    time_grain = st.selectbox("Time grain", TIME_GRAINS, key="v4_time_grain")
    time_settings = get_v3_time_settings(time_grain)

    if not DEFAULT_DATA_PATH.exists():
        st.error("Default modeling workbook was not found.")
        return
    try:
        base = filter_rolled_up_products(auto_load_modeling(str(DEFAULT_DATA_PATH), time_grain=time_grain))
    except Exception as exc:
        st.error(f"Unable to load default workbook: {exc}")
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
        st.warning(f"No states have enough {time_grain.lower()} rolled-up workbook data for V4.")
        return

    c1, c2 = st.columns(2)
    with c1:
        state = st.selectbox("State", eligible_states, key="v4_state")
    with c2:
        product = st.selectbox("Rolled-up product", get_available_rolled_up_products(base, state), key="v4_product")
    filtered = base[base["STATE_CD"] == state].copy()
    if product != PRODUCT_ALL_LABEL:
        filtered = filtered[filtered["PRODUCT_CD"] == product].copy()
    filtered = aggregate_v4_filtered_data(filtered, time_grain=time_grain)

    if filtered.empty:
        st.warning("No rolled-up product data for the selected V4 state/product.")
        return

    selected_train_rows = int(filtered.loc[filtered["ISO_YEAR"].isin(TRAIN_YEARS_V3)].shape[0])
    selected_test_rows = int(filtered.loc[filtered["ISO_YEAR"].isin(TEST_YEARS_V3)].shape[0])
    st.info(
        f"**Pipeline:** train on {TRAIN_YEARS_V3} · test on {TEST_YEARS_V3} · "
        f"seasonality via {time_settings['seasonality_label']} · "
        f"rolling 2026 {time_settings['test_unit_label']} · "
        f"rolled-up products only · re-aggregated to year-period-channel · simple regression only · "
        f"selected state train rows = {selected_train_rows}, test rows = {selected_test_rows}"
    )

    for channel in CHANNELS:
        st.subheader(f"{channel} channel")
        channel_train_rows = int(
            filtered.loc[
                (filtered["CHANNEL_CD"] == channel) & (filtered["ISO_YEAR"].isin(TRAIN_YEARS_V3))
            ].shape[0]
        )
        if channel_train_rows < 20:
            st.warning(
                f"{channel} has only {channel_train_rows} rolled-up training rows after re-aggregation. "
                "V4 needs at least 20 train rows."
            )
            continue
        with st.spinner(f"Running V4 regression for {channel}…"):
            result = run_workbook_rolling_regression(
                filtered,
                channel,
                time_grain=time_grain,
                train_years=TRAIN_YEARS_V3,
                test_years=TEST_YEARS_V3,
                min_train_rows=20,
            )

        if result is None:
            st.info(f"Insufficient rolled-up data for {channel} modeling under the V4 split.")
            continue

        avg_apps = result["avg_test_apps"]
        render_mmm_rolling_result(result, avg_apps, key_prefix=f"v4-{time_grain.lower()}-{channel.lower()}")
        st.divider()


# ---------------------------------------------------------------------------
# Version 5 — Live Model Diagnostics with Offline Validation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading offline diagnostics…")
def load_offline_diagnostics(channel: str) -> pd.DataFrame:
    path = DIAGNOSTICS_DIGITAL_PATH if channel == "DIGITAL" else DIAGNOSTICS_PHYSICAL_PATH
    df = pd.read_excel(str(path), sheet_name="in", engine="openpyxl")
    for col in ["R2", "AdjR2", "MAE", "MAPE", "RMSE", "Test_R2", "AIC", "BIC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _v5_config_label(row) -> str:
    return f"{row['model_type']} | {row['dummy_family']}"


def render_tab_mmm_v5() -> None:
    render_version_intro(
        "Version 5 — Live Regression Pipeline with Offline Validation",
        [
            "Rebuilds the complete modeling frame from raw files "
            "(`Marketing_Spend_Data.csv`, `Originations_Data1_1.xlsx`, `DM_Data.csv`) — "
            "no pre-built workbook needed.",
            "The target variable is `NON_DM_APPLICATIONS = max(0, APPLICATIONS − DM_APPLICATIONS)` "
            "— spend-driven applications only, with Direct Mail responses subtracted.",
            "Models are run at **state scope** (each state independently) or **division scope** "
            "(states pooled into 9 US Census Divisions).",
            "Six configurations per entity: {NNLS, OLS} × {Fortnightly dummies F_1..F_25, "
            "Weekly dummies W_2..W_52, Fourier terms sin/cos k=1,2}. "
            "NNLS/f_dummy and NNLS/weekly use MinMax scaler; fourier uses Standard scaler.",
            "Train: 2024 + 2025 data (104 rows per state). "
            "Test: first 8 weeks of 2026 (8 rows per state, 32 per 4-state division).",
            "**Validation tab** lets you compare live results to the pre-computed "
            "`consolidated_model_diagnostics_*.xlsx` files — "
            "R², AdjR², AIC, BIC, and Test_R² should match exactly.",
        ],
        note=(
            "Sweepstakes is excluded from all models. "
            "DM split: ONLINE channel = DIGITAL DM; OMNI + STORE = PHYSICAL DM. "
            "Scaler is fit on training data only and applied to test."
        ),
    )

    # ------------------------------------------------------------------
    # Filter row — channel / scope / entity / run button
    # ------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    with c1:
        v5_channel = st.selectbox("Channel", ["DIGITAL", "PHYSICAL"], key="v5_channel")
    with c2:
        v5_scope = st.selectbox("Scope", ["state", "division"], key="v5_scope")

    # Load frame to get entity list (cached)
    from utils import MARKETING_SPEND_PATH as _msp, DM_DATA_PATH as _dmp, ORIGINATIONS_V5_PATH as _op
    missing_files = []
    if not _msp.exists():
        missing_files.append(str(_msp.name))
    if not _dmp.exists():
        missing_files.append(str(_dmp.name))
    if not _op.exists():
        missing_files.append(str(_op.name))
    if missing_files:
        st.error(f"Missing raw data files: {', '.join(missing_files)}. Please add them to the app directory.")
        return

    try:
        frame_df = cached_build_modeling_frame(v5_channel)
    except Exception as exc:
        st.error(f"Failed to build modeling frame: {exc}")
        return

    if v5_scope == "state":
        entity_list = sorted(frame_df["STATE_CD"].dropna().unique().tolist())
    else:
        entity_list = sorted(frame_df["Division"].dropna().unique().tolist())

    with c3:
        v5_entity = st.selectbox("Entity", entity_list, key="v5_entity")
    with c4:
        st.write("")  # spacer to align button
        v5_run = st.button("▶ Run All 6 Configurations", key="v5_run")

    cache_key = (v5_channel, v5_scope, v5_entity)
    if v5_run:
        with st.spinner(f"Running all 6 configs for {v5_entity} ({v5_channel})…"):
            diag_df = run_all_configs_for_entity(frame_df, v5_scope, v5_entity, v5_channel)
        if "v5_results" not in st.session_state:
            st.session_state["v5_results"] = {}
        st.session_state["v5_results"][cache_key] = diag_df

    results: pd.DataFrame | None = (
        st.session_state.get("v5_results", {}).get(cache_key)
    )

    if results is None or results.empty:
        st.info("Select channel, scope, and entity, then click **▶ Run All 6 Configurations** to run the pipeline.")
        return

    # Separate display columns from _result column
    display_cols = [
        "model_type", "dummy_family", "scaler_type", "train_rows", "test_rows",
        "R2", "AdjR2", "MAPE", "Test_R2", "AIC", "BIC",
    ]
    disp_df = results[display_cols].copy()

    # Compute best config
    valid = results[results["Test_R2"] > 0].copy()
    best_idx = None
    if not valid.empty:
        valid["_score"] = (
            valid["AdjR2"].rank(ascending=False)
            + valid["MAPE"].rank(ascending=True)
            + valid["AIC"].rank(ascending=True)
        )
        best_idx = int(valid["_score"].idxmin())

    config_labels = [_v5_config_label(results.iloc[i]) for i in range(len(results))]

    # ------------------------------------------------------------------
    # Sub-tabs
    # ------------------------------------------------------------------
    sub_tabs = st.tabs(["📊 Model Comparison", "📈 Actual vs Predicted", "🔍 Coefficient Explorer", "✅ Validate vs Offline"])

    # ---- Sub-tab 1: Model Comparison ----------------------------------
    with sub_tabs[0]:
        st.subheader("All 6 Configurations")

        label_col = disp_df.copy()
        label_col.insert(0, "Config", config_labels)
        if best_idx is not None:
            label_col["Best"] = ""
            label_col.at[best_idx, "Best"] = "⭐ Best"

        st.dataframe(
            label_col,
            use_container_width=True,
            column_config={
                "R2": st.column_config.ProgressColumn("R²", min_value=0, max_value=1, format="%.4f"),
                "AdjR2": st.column_config.ProgressColumn("Adj R²", min_value=0, max_value=1, format="%.4f"),
                "Test_R2": st.column_config.ProgressColumn("Test R²", min_value=0, max_value=1, format="%.4f"),
                "MAPE": st.column_config.NumberColumn("MAPE (%)", format="%.2f%%"),
                "AIC": st.column_config.NumberColumn("AIC", format="%.2f"),
                "BIC": st.column_config.NumberColumn("BIC", format="%.2f"),
            },
            hide_index=True,
        )

        # Metric cards for best config
        if best_idx is not None:
            best = results.iloc[best_idx]
            st.markdown("**Best Configuration**")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Model Type", best["model_type"])
            mc2.metric("Dummy Family", best["dummy_family"])
            mc3.metric("Adj R²", f"{best['AdjR2']:.4f}")
            mc4.metric("Test R²", f"{best['Test_R2']:.4f}")

        # Grouped bar chart: AdjR2 + MAPE
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="Adj R²",
            x=config_labels,
            y=results["AdjR2"].tolist(),
            yaxis="y",
            marker_color="#4C78A8",
        ))
        fig_cmp.add_trace(go.Bar(
            name="MAPE (%)",
            x=config_labels,
            y=results["MAPE"].tolist(),
            yaxis="y2",
            marker_color="#F58518",
        ))
        fig_cmp.update_layout(
            title="Adj R² vs MAPE by Configuration",
            yaxis=dict(title="Adj R²", range=[0, 1]),
            yaxis2=dict(title="MAPE (%)", overlaying="y", side="right"),
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400,
        )
        st.plotly_chart(fig_cmp, use_container_width=True, key="v5_cmp_chart")

    # ---- Sub-tab 2: Actual vs Predicted -------------------------------
    with sub_tabs[1]:
        sel_label_2 = st.selectbox(
            "Configuration", config_labels, key="v5_sel_config_2"
        )
        sel_idx_2 = config_labels.index(sel_label_2)
        res2 = results.iloc[sel_idx_2]["_result"]

        # Build period labels
        train_periods = res2["train_periods"]
        test_periods = res2["test_periods"]
        train_labels = [f"{int(r[0])}-W{int(r[1]):02d}" for r in train_periods]
        test_labels = [f"{int(r[0])}-W{int(r[1]):02d}" for r in test_periods]
        all_labels = train_labels + test_labels
        all_actual = np.concatenate([res2["y_train_actual"], res2["y_test_actual"]])
        all_pred = np.concatenate([res2["y_train_pred"], res2["y_test_pred"]])
        n_train = len(train_labels)

        # Actual vs Predicted chart — use integer x-axis so add_vline works correctly
        n_tr = len(train_labels)
        n_te = len(test_labels)
        x_tr = list(range(n_tr))
        x_te = list(range(n_tr, n_tr + n_te))
        x_all = list(range(n_tr + n_te))

        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(
            x=x_tr, y=res2["y_train_actual"].tolist(),
            mode="lines", name="Actual (Train)", line=dict(color="#4C78A8", width=2),
        ))
        fig_avp.add_trace(go.Scatter(
            x=x_tr, y=res2["y_train_pred"].tolist(),
            mode="lines", name="Predicted (Train)", line=dict(color="#4C78A8", width=2, dash="dash"),
        ))
        fig_avp.add_trace(go.Scatter(
            x=x_te, y=res2["y_test_actual"].tolist(),
            mode="lines", name="Actual (Test)", line=dict(color="#F58518", width=2),
        ))
        fig_avp.add_trace(go.Scatter(
            x=x_te, y=res2["y_test_pred"].tolist(),
            mode="lines", name="Predicted (Test)", line=dict(color="#F58518", width=2, dash="dash"),
        ))
        if n_tr > 0:
            fig_avp.add_vline(
                x=n_tr - 0.5, line_dash="dot", line_color="gray",
                annotation_text="Train | Test", annotation_position="top right",
            )
        fig_avp.update_layout(
            title=f"Actual vs Predicted — {sel_label_2}",
            xaxis=dict(
                title="Period",
                tickmode="array",
                tickvals=x_all,
                ticktext=all_labels,
                tickangle=-90,
            ),
            yaxis_title="NON_DM_APPLICATIONS",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_avp, use_container_width=True, key="v5_avp_chart")

        # Residuals chart
        residuals = all_actual - all_pred
        colors = ["#E45756" if r < 0 else "#4C78A8" for r in residuals]
        fig_res = go.Figure(go.Bar(x=x_all, y=residuals.tolist(), marker_color=colors))
        fig_res.add_hline(y=0, line_color="black", line_width=1)
        if n_tr > 0:
            fig_res.add_vline(x=n_tr - 0.5, line_dash="dot", line_color="gray")
        fig_res.update_layout(
            title="Residuals (Actual − Predicted)",
            xaxis=dict(
                title="Period",
                tickmode="array",
                tickvals=x_all,
                ticktext=all_labels,
                tickangle=-90,
            ),
            yaxis_title="Residual",
            height=300,
        )
        st.plotly_chart(fig_res, use_container_width=True, key="v5_res_chart")

        # Metrics
        from sklearn.metrics import r2_score as _r2, mean_absolute_error as _mae
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Train R²", f"{results.iloc[sel_idx_2]['R2']:.4f}")
        m2.metric("Train MAE", f"{results.iloc[sel_idx_2]['MAE']:.2f}")
        m3.metric("Train MAPE", f"{results.iloc[sel_idx_2]['MAPE']:.2f}%")
        m4.metric("Test R²", f"{results.iloc[sel_idx_2]['Test_R2']:.4f}")
        te_actual = res2["y_test_actual"]
        te_pred = res2["y_test_pred"]
        if len(te_actual) > 0:
            test_mae = float(_mae(te_actual, te_pred))
            te_mask = te_actual != 0
            test_mape = (
                float(np.mean(np.abs((te_actual[te_mask] - te_pred[te_mask]) / te_actual[te_mask])) * 100)
                if te_mask.any() else float("nan")
            )
        else:
            test_mae = float("nan")
            test_mape = float("nan")
        m5.metric("Test MAE", f"{test_mae:.2f}" if not np.isnan(test_mae) else "N/A")
        m6.metric("Test MAPE", f"{test_mape:.2f}%" if not np.isnan(test_mape) else "N/A")

    # ---- Sub-tab 3: Coefficient Explorer ------------------------------
    with sub_tabs[2]:
        sel_label_3 = st.selectbox(
            "Configuration", config_labels, key="v5_sel_config_3"
        )
        sel_idx_3 = config_labels.index(sel_label_3)
        res3 = results.iloc[sel_idx_3]["_result"]
        feats = res3["features"]
        coefs = res3["coefs"]

        from data_processing import TACTIC_COLS_V5 as _tactics
        feat_types = ["tactic" if f in _tactics else "seasonal" for f in feats]
        coef_colors = ["#4C78A8" if t == "tactic" else "#72B7B2" for t in feat_types]

        sorted_pairs = sorted(zip(feats, coefs, coef_colors), key=lambda x: abs(x[1]), reverse=True)
        s_feats, s_coefs, s_colors = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

        fig_coef = go.Figure(go.Bar(
            x=list(s_coefs),
            y=list(s_feats),
            orientation="h",
            marker_color=list(s_colors),
        ))
        fig_coef.update_layout(
            title=f"Coefficients — {sel_label_3}",
            xaxis_title="Coefficient Value",
            yaxis_title="Feature",
            height=max(400, len(feats) * 22),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_coef, use_container_width=True, key="v5_coef_chart")

        coef_table = pd.DataFrame({
            "Feature": feats,
            "Coefficient": [round(float(c), 6) for c in coefs],
            "Type": feat_types,
        })
        st.dataframe(coef_table, use_container_width=True, hide_index=True)

        model_type_3 = results.iloc[sel_idx_3]["model_type"]
        if model_type_3 == "NNLS":
            st.caption(
                "For NNLS models, coefficients are constrained to be non-negative (no suppression effects). "
                "Each coefficient represents the change in NON_DM_APPLICATIONS per unit change in the scaled predictor."
            )
        else:
            st.caption(
                "For OLS models, coefficients may be negative. "
                "Each coefficient represents the change in NON_DM_APPLICATIONS per unit change in the scaled predictor."
            )

    # ---- Sub-tab 4: Validate vs Offline --------------------------------
    with sub_tabs[3]:
        diag_path = DIAGNOSTICS_DIGITAL_PATH if v5_channel == "DIGITAL" else DIAGNOSTICS_PHYSICAL_PATH
        if not diag_path.exists():
            st.warning(
                "Offline diagnostics file not found. Place "
                "consolidated_model_diagnostics_digital.xlsx / "
                "consolidated_model_diagnostics_physical.xlsx in the app directory to enable validation."
            )
        else:
            try:
                offline_df = load_offline_diagnostics(v5_channel)
            except Exception as exc:
                st.error(f"Failed to load offline diagnostics: {exc}")
                return

            metric_cols = ["R2", "AdjR2", "Test_R2", "AIC", "BIC"]
            match_count = 0
            comparison_rows = []

            for _, live_row in results.iterrows():
                mt = live_row["model_type"]
                df_fam = live_row["dummy_family"]
                ent = live_row["entity"]

                # Match in offline file
                mask = (
                    (offline_df.get("entity", pd.Series(dtype=str)) == ent)
                    & (offline_df.get("model_type", pd.Series(dtype=str)) == mt)
                    & (offline_df.get("dummy_family", pd.Series(dtype=str)) == df_fam)
                )
                matched = offline_df[mask]

                config = _v5_config_label(live_row)
                row_data: dict = {"Config": config}
                all_match = True
                for m in metric_cols:
                    live_val = live_row.get(m, np.nan)
                    if not matched.empty and m in matched.columns:
                        off_val = float(matched.iloc[0][m])
                    else:
                        off_val = np.nan
                    diff = abs(live_val - off_val) if not (np.isnan(live_val) or np.isnan(off_val)) else np.nan
                    row_data[f"{m}_live"] = round(float(live_val), 6) if not np.isnan(live_val) else np.nan
                    row_data[f"{m}_offline"] = round(float(off_val), 6) if not np.isnan(off_val) else np.nan
                    row_data[f"{m}_diff"] = round(float(diff), 6) if diff is not None and not np.isnan(diff) else np.nan
                    if diff is None or np.isnan(diff) or diff >= 0.01:
                        all_match = False
                if all_match:
                    match_count += 1
                comparison_rows.append(row_data)

            cmp_df = pd.DataFrame(comparison_rows)

            # Summary message
            total = len(results)
            if match_count == total:
                st.success(f"✅ {match_count} of {total} configurations matched within tolerance (|diff| < 0.01).")
            else:
                st.warning(
                    f"⚠ {total - match_count} configuration(s) differ — check data versions. "
                    f"{match_count} of {total} matched within tolerance."
                )

            # Side-by-side comparison
            st.subheader("Live vs Offline Diagnostics")
            for m in metric_cols:
                with st.expander(f"**{m}** — live / offline / diff", expanded=(m in ["R2", "Test_R2"])):
                    sub = cmp_df[["Config", f"{m}_live", f"{m}_offline", f"{m}_diff"]].copy()
                    sub.columns = ["Config", "Live", "Offline", "Diff"]

                    def _color_diff(val):
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            return "color: gray"
                        if abs(val) < 0.001:
                            return "color: green"
                        if abs(val) < 0.01:
                            return "color: orange"
                        return "color: red"

                    styled = sub.style.applymap(_color_diff, subset=["Diff"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

            st.caption(
                "R², AdjR², AIC, BIC, and Test_R² are expected to match exactly. "
                "MAE/MAPE/RMSE may differ slightly due to how residuals are computed in the offline notebook."
            )


# ---------------------------------------------------------------------------
# Version 6 — 8-Iteration OLS Pipeline (Prescreen Variants)
# ---------------------------------------------------------------------------

_V6_FIXED_STATES = ["AL", "CA", "DE", "FL"]
_V6_CHANNELS = ["DIGITAL", "PHYSICAL"]


def render_tab_mmm_v6() -> None:
    render_version_intro(
        "Version 6 — OLS Regression Diagnostic Table",
        [
            "Fixed scope: states AL, CA, DE, FL × channels DIGITAL, PHYSICAL × 16 iterations = 128 configurations.",
            "**No NNLS, no Fourier.** OLS only. Target: NON_DM_APPLICATIONS.",
            "Train: 2024 + 2025. OOS test: first 8 weeks of 2026.",
        ],
        note="Uses the NON-DM workflow. Target = APPLICATIONS - DM_APPLICATIONS, clipped at zero. Sweepstakes excluded. Lag applied per state. Interaction computed on scaled values.",
    )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    from utils import (
        MARKETING_SPEND_PATH as _msp6, ORIGINATIONS_V5_PATH as _op6,
    )
    from data_processing import DM_DIFF_PATH as _dm_diff6, fit_v6_iteration as _fit_v6, TACTIC_COLS_V5 as _v6_tactics

    missing = [n for p, n in [(_msp6, _msp6.name), (_op6, _op6.name), (_dm_diff6, _dm_diff6.name)] if not p.exists()]
    if missing:
        st.error(f"Missing raw data files: {', '.join(missing)}")
        return

    v6_run = st.button("▶ Run All", key="v6_run")

    if v6_run:
        all_rows = []
        with st.spinner("Running 128 configurations (16 iterations × AL/CA/DE/FL × DIGITAL/PHYSICAL)…"):
            for _ch in ["DIGITAL", "PHYSICAL"]:
                try:
                    _frame = cached_build_v7_modeling_frame(_ch)
                except Exception as exc:
                    st.warning(f"Could not build NON-DM frame for {_ch}: {exc}")
                    continue
                for _state in _V6_FIXED_STATES:
                    _edf = _frame[_frame["STATE_CD"] == _state].copy()
                    _edf = _edf.sort_values(["ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)
                    for _cfg in V6_ITERATIONS:
                        _res = _call_fit_v6_compat(_fit_v6, _edf, _ch, _cfg)
                        if _res is None:
                            continue
                        _yte = _res["y_test_actual"]
                        _ytp = _res["y_test_pred"]
                        _oos_rmse = float(np.sqrt(np.mean((_yte - _ytp) ** 2))) if len(_yte) > 0 else np.nan
                        _tac_coefs = [
                            (f, float(c))
                            for f, c in zip(_res["features"], _res["coefs"])
                            if f in _res.get("tactic_cols", [])
                        ]
                        _coef_str = " | ".join(f"{f}: {c:.3f}" for f, c in _tac_coefs)
                        all_rows.append({
                            "iter_num": _cfg["num"],
                            "Iteration": _cfg["label"],
                            "_state": _state,
                            "_channel": _ch,
                            "MAPE": round(float(_res["MAPE"]), 2),
                            "R. Sq": round(float(_res["R2"]), 4),
                            "RMSE": round(float(_res["RMSE"]), 2),
                            "OOS RMSE": round(_oos_rmse, 2) if not np.isnan(_oos_rmse) else np.nan,
                            "Coefficients": _coef_str,
                            "_result": _res,
                        })
        st.session_state["v6_all"] = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    all_results: pd.DataFrame | None = st.session_state.get("v6_all")
    if all_results is None or all_results.empty:
        st.info("Click **▶ Run All** to compute all 64 configurations.")
        return

    sub_tabs = st.tabs([
        "📊 Diagnostic Table",
        "📈 Actual vs Predicted",
        "🔍 Coefficients",
        "✅ Validate vs Offline",
    ])

    # ---- Sub-tab 1: Hierarchical Diagnostic Table ---------------------
    with sub_tabs[0]:
        _viz_df = all_results.copy()
        _viz_df["combo"] = _viz_df["_state"] + " | " + _viz_df["_channel"]

        _best_idx = (
            _viz_df.sort_values(["MAPE", "R. Sq"], ascending=[True, False])
            .groupby(["_state", "_channel"], dropna=False)
            .head(1)
            .index
        )
        _best_combo_df = (
            _viz_df.loc[_best_idx, ["_state", "_channel", "iter_num", "MAPE", "R. Sq"]]
            .sort_values(["_state", "_channel"])
            .rename(columns={
                "_state": "State",
                "_channel": "Channel",
                "iter_num": "Best Iteration",
            })
            .reset_index(drop=True)
        )

        _iter_summary = (
            _viz_df.groupby(["iter_num", "Iteration"], dropna=False)
            .agg(
                avg_mape=("MAPE", "mean"),
                avg_rsq=("R. Sq", "mean"),
            )
            .reset_index()
            .sort_values(["avg_mape", "avg_rsq"], ascending=[True, False])
        )
        _best_overall = _iter_summary.iloc[0]

        _channel_summary = (
            _viz_df.groupby(["_channel", "iter_num", "Iteration"], dropna=False)
            .agg(
                avg_mape=("MAPE", "mean"),
                avg_rsq=("R. Sq", "mean"),
            )
            .reset_index()
            .sort_values(["_channel", "avg_mape", "avg_rsq"], ascending=[True, True, False])
        )
        _best_digital = _channel_summary[_channel_summary["_channel"] == "DIGITAL"].iloc[0]
        _best_physical = _channel_summary[_channel_summary["_channel"] == "PHYSICAL"].iloc[0]

        _state_summary = (
            _viz_df.groupby("_state", dropna=False)
            .agg(
                avg_mape=("MAPE", "mean"),
                avg_rsq=("R. Sq", "mean"),
            )
            .reset_index()
            .sort_values(["avg_mape", "avg_rsq"], ascending=[True, False])
        )
        _best_state = _state_summary.iloc[0]

        st.markdown("**Quick Read**")
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Best Overall Iteration", str(int(_best_overall["iter_num"])))
        qc2.metric("Best DIGITAL Iteration", str(int(_best_digital["iter_num"])))
        qc3.metric("Best PHYSICAL Iteration", str(int(_best_physical["iter_num"])))
        qc4.metric("Best State Avg MAPE", f"{_best_state['_state']} ({_best_state['avg_mape']:.2f}%)")

        st.caption(
            "Lower MAPE is better. Higher R² is better. "
            "These visuals summarize which iterations are strongest before the detailed table."
        )

        vc1, vc2 = st.columns([1.1, 1])
        with vc1:
            _heat_source = _viz_df.copy()
            _heat_source["iter_num_label"] = _heat_source["iter_num"].astype(int).astype(str)
            _heat = (
                _heat_source.pivot_table(
                    index="combo",
                    columns="iter_num_label",
                    values="MAPE",
                    aggfunc="mean",
                )
                .sort_index()
            )
            _heat = _heat.reindex(columns=[str(cfg["num"]) for cfg in V6_ITERATIONS if str(cfg["num"]) in _heat.columns])
            fig_heat_v6 = go.Figure(data=go.Heatmap(
                z=_heat.values,
                x=_heat.columns.tolist(),
                y=_heat.index.tolist(),
                colorscale="YlGnBu_r",
                colorbar_title="MAPE (%)",
                hovertemplate="State|Channel: %{y}<br>Iteration: %{x}<br>MAPE: %{z:.2f}%<extra></extra>",
            ))
            fig_heat_v6.update_layout(
                title="MAPE Heatmap by State and Channel",
                height=max(380, 34 * len(_heat.index)),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_heat_v6, use_container_width=True, key="v6_heatmap_summary")

        with vc2:
            _bar_df = _channel_summary.copy()
            _bar_df["iter_num_label"] = _bar_df["iter_num"].astype(int).astype(str)
            fig_bar_v6 = px.bar(
                _bar_df,
                x="iter_num_label",
                y="avg_rsq",
                color="_channel",
                barmode="group",
                color_discrete_map={"DIGITAL": "#4C78A8", "PHYSICAL": "#F58518"},
                labels={
                    "iter_num_label": "Iteration",
                    "avg_rsq": "Average R²",
                    "_channel": "Channel",
                },
                title="Average R² by Iteration and Channel",
            )
            fig_bar_v6.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=380)
            st.plotly_chart(fig_bar_v6, use_container_width=True, key="v6_channel_summary")

            st.markdown("**Best Iteration by State and Channel**")
            st.dataframe(
                _best_combo_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "MAPE": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
                    "R. Sq": st.column_config.NumberColumn("R²", format="%.4f"),
                },
            )

        st.divider()

        # Build sparse hierarchical rows: Iteration > State > Channel
        _ch_display = {"PHYSICAL": "Physical", "DIGITAL": "Digital"}
        _table_rows = []
        for _cfg in V6_ITERATIONS:
            _first_iter = True
            for _state in _V6_FIXED_STATES:
                _first_state = True
                for _ch in ["PHYSICAL", "DIGITAL"]:
                    _m = all_results[
                        (all_results["iter_num"] == _cfg["num"]) &
                        (all_results["_state"] == _state) &
                        (all_results["_channel"] == _ch)
                    ]
                    if _m.empty:
                        _mape, _rsq, _rmse, _oos, _coef = np.nan, np.nan, np.nan, np.nan, ""
                    else:
                        _r = _m.iloc[0]
                        _mape, _rsq, _rmse, _oos, _coef = (
                            _r["MAPE"], _r["R. Sq"], _r["RMSE"], _r["OOS RMSE"], _r["Coefficients"]
                        )
                    _table_rows.append({
                        "Iteration #": str(_cfg["num"]) if _first_iter else "",
                        "Iteration": _cfg["label"] if _first_iter else "",
                        "State":     _state if _first_state else "",
                        "Channel":   _ch_display[_ch],
                        "MAPE":      _mape,
                        "R. Sq":      _rsq,
                        "RMSE":      _rmse,
                        "OOS RMSE":  _oos,
                        "Coefficients": _coef,
                    })
                    _first_iter = False
                    _first_state = False

        _tbl_df = pd.DataFrame(_table_rows)
        st.dataframe(
            _tbl_df,
            use_container_width=True,
            height=min(35 * len(_table_rows) + 38, 900),
            column_config={
                "Iteration #":   st.column_config.TextColumn("Iteration #", width="small"),
                "Iteration":    st.column_config.TextColumn("Iteration", width="large"),
                "State":        st.column_config.TextColumn("State",     width="small"),
                "Channel":      st.column_config.TextColumn("Channel",   width="small"),
                "MAPE":         st.column_config.NumberColumn("MAPE (%)",  format="%.2f"),
                "R. Sq":         st.column_config.NumberColumn("R²",        format="%.4f"),
                "RMSE":         st.column_config.NumberColumn("RMSE",      format="%.2f"),
                "OOS RMSE":     st.column_config.NumberColumn("OOS RMSE",  format="%.2f"),
                "Coefficients": st.column_config.TextColumn("Coefficients", width="large"),
            },
            hide_index=True,
        )
        st.caption(
            "The table groups rows by Iteration # → Iteration → State → Channel. "
            "When a cell in Iteration #, Iteration, or State is blank, it belongs to the same group as the last non-blank value above it. "
            "OOS RMSE = prediction error on the first 8 weeks of 2026 (data the model never trained on). "
            "Lower MAPE/RMSE/OOS RMSE = better fit. Higher R² = more variance explained."
        )

    # ---- Sub-tab 2: Actual vs Predicted -------------------------------
    with sub_tabs[1]:
        _avp_iters = [cfg["label"] for cfg in V6_ITERATIONS]
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            sel_state_avp = st.selectbox("State", _V6_FIXED_STATES, key="v6_avp_state")
        with ac2:
            sel_ch_avp = st.selectbox("Channel", ["DIGITAL", "PHYSICAL"], key="v6_avp_ch")
        with ac3:
            sel_iter_avp = st.selectbox("Iteration", _avp_iters, key="v6_avp_iter")

        _match_avp = all_results[
            (all_results["_state"] == sel_state_avp) &
            (all_results["_channel"] == sel_ch_avp) &
            (all_results["Iteration"] == sel_iter_avp)
        ]
        if _match_avp.empty:
            st.warning("No result for this selection.")
        else:
            res_avp = _match_avp.iloc[0]["_result"]
            tr_labels = [f"{int(r[0])}-W{int(r[1]):02d}" for r in res_avp["train_periods"]]
            te_labels = [f"{int(r[0])}-W{int(r[1]):02d}" for r in res_avp["test_periods"]]
            all_labels = tr_labels + te_labels
            n_tr, n_te = len(tr_labels), len(te_labels)
            x_tr = list(range(n_tr))
            x_te = list(range(n_tr, n_tr + n_te))
            x_all = list(range(n_tr + n_te))

            fig_avp6 = go.Figure()
            fig_avp6.add_trace(go.Scatter(x=x_tr, y=res_avp["y_train_actual"].tolist(),
                mode="lines", name="Actual (Train)", line=dict(color="#4C78A8", width=2)))
            fig_avp6.add_trace(go.Scatter(x=x_tr, y=res_avp["y_train_pred"].tolist(),
                mode="lines", name="Predicted (Train)", line=dict(color="#4C78A8", width=2, dash="dash")))
            fig_avp6.add_trace(go.Scatter(x=x_te, y=res_avp["y_test_actual"].tolist(),
                mode="lines", name="Actual (Test)", line=dict(color="#F58518", width=2)))
            fig_avp6.add_trace(go.Scatter(x=x_te, y=res_avp["y_test_pred"].tolist(),
                mode="lines", name="Predicted (Test)", line=dict(color="#F58518", width=2, dash="dash")))
            if n_tr > 0:
                fig_avp6.add_vline(x=n_tr - 0.5, line_dash="dot", line_color="gray",
                                   annotation_text="Train | Test", annotation_position="top right")
            fig_avp6.update_layout(
                title=f"Actual vs Predicted — {sel_state_avp} {sel_ch_avp}",
                xaxis=dict(title="Period", tickmode="array", tickvals=x_all,
                           ticktext=all_labels, tickangle=-90),
                yaxis_title="NON_DM_APPLICATIONS", height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_avp6, use_container_width=True, key="v6_avp_chart")

            residuals6 = (
                np.concatenate([res_avp["y_train_actual"], res_avp["y_test_actual"]]) -
                np.concatenate([res_avp["y_train_pred"], res_avp["y_test_pred"]])
            )
            colors6 = ["#E45756" if r < 0 else "#4C78A8" for r in residuals6]
            fig_res6 = go.Figure(go.Bar(x=x_all, y=residuals6.tolist(), marker_color=colors6))
            fig_res6.add_hline(y=0, line_color="black", line_width=1)
            if n_tr > 0:
                fig_res6.add_vline(x=n_tr - 0.5, line_dash="dot", line_color="gray")
            fig_res6.update_layout(
                title="Residuals",
                xaxis=dict(tickmode="array", tickvals=x_all, ticktext=all_labels, tickangle=-90),
                yaxis_title="Residual", height=280,
            )
            st.plotly_chart(fig_res6, use_container_width=True, key="v6_res_chart")

            _row6 = _match_avp.iloc[0]
            _te_mask6 = res_avp["y_test_actual"] != 0
            _te_mape6 = (
                float(np.mean(np.abs(
                    (res_avp["y_test_actual"][_te_mask6] - res_avp["y_test_pred"][_te_mask6])
                    / res_avp["y_test_actual"][_te_mask6]
                )) * 100) if _te_mask6.any() else float("nan")
            )
            _train_rsq6 = _row6.get("R. Sq", _row6.get("R.Sq", _row6.get("R2", np.nan)))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train R²",   f"{_train_rsq6:.4f}" if not np.isnan(_train_rsq6) else "N/A")
            m2.metric("Train MAPE", f"{_row6['MAPE']:.2f}%")
            m3.metric("OOS RMSE",   f"{_row6['OOS RMSE']:.2f}" if not np.isnan(_row6["OOS RMSE"]) else "N/A")
            m4.metric("OOS MAPE",   f"{_te_mape6:.2f}%" if not np.isnan(_te_mape6) else "N/A")

    # ---- Sub-tab 3: Coefficients --------------------------------------
    with sub_tabs[2]:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            sel_state_coef = st.selectbox("State", _V6_FIXED_STATES, key="v6_coef_state")
        with cc2:
            sel_ch_coef = st.selectbox("Channel", ["DIGITAL", "PHYSICAL"], key="v6_coef_ch")
        with cc3:
            sel_iter_coef = st.selectbox("Iteration", [cfg["label"] for cfg in V6_ITERATIONS], key="v6_coef_iter")

        _match_coef = all_results[
            (all_results["_state"] == sel_state_coef) &
            (all_results["_channel"] == sel_ch_coef) &
            (all_results["Iteration"] == sel_iter_coef)
        ]
        if _match_coef.empty:
            st.warning("No result for this selection.")
        else:
            res_coef = _match_coef.iloc[0]["_result"]
            feats6 = res_coef["features"]
            coefs6 = res_coef["coefs"]
            types6 = ["tactic" if f in _v6_tactics else "seasonal" for f in feats6]
            colors_coef = ["#4C78A8" if t == "tactic" else "#72B7B2" for t in types6]
            sorted_pairs6 = sorted(zip(feats6, coefs6, colors_coef), key=lambda x: abs(x[1]), reverse=True)
            sf, sc, scol = zip(*sorted_pairs6) if sorted_pairs6 else ([], [], [])
            if sf:
                fig_coef6 = go.Figure(go.Bar(
                    x=list(sc), y=list(sf), orientation="h", marker_color=list(scol),
                ))
                fig_coef6.update_layout(
                    title=f"Coefficients — {sel_state_coef} {sel_ch_coef}",
                    xaxis_title="Coefficient Value", yaxis_title="Feature",
                    height=max(400, len(feats6) * 22), yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_coef6, use_container_width=True, key="v6_coef_chart")
                coef_tbl6 = pd.DataFrame({
                    "Feature": feats6,
                    "Coefficient": [round(float(c), 6) for c in coefs6],
                    "Type": types6,
                })
                st.dataframe(coef_tbl6, use_container_width=True, hide_index=True)
                st.caption("OLS coefficients on MinMax-scaled tactic predictors.")

    # ---- Sub-tab 4: Validate vs Offline --------------------------------
    with sub_tabs[3]:
        st.markdown("Iterations 1 and 2 can be validated against pre-computed offline diagnostics.")
        _val_ch = st.selectbox("Channel", ["DIGITAL", "PHYSICAL"], key="v6_val_ch")
        _val_state = st.selectbox("State", _V6_FIXED_STATES, key="v6_val_state")

        _diag_path6 = DIAGNOSTICS_DIGITAL_PATH if _val_ch == "DIGITAL" else DIAGNOSTICS_PHYSICAL_PATH
        if not _diag_path6.exists():
            st.warning("Offline diagnostics file not found.")
        else:
            try:
                _offline6 = load_offline_diagnostics(_val_ch)
            except Exception as exc:
                st.error(f"Failed to load offline diagnostics: {exc}")
                return
            _baseline_map = {1: "weekly", 2: "f_dummy"}
            _metric_cols6 = ["R2", "AdjR2", "Test_R2", "AIC", "BIC"]
            _cmp_rows6, _match_count6 = [], 0
            for _iter_num, _df_fam in _baseline_map.items():
                _iter_label = next((c["label"] for c in V6_ITERATIONS if c["num"] == _iter_num), "")
                _live_rows = all_results[
                    (all_results["iter_num"] == _iter_num) &
                    (all_results["_state"] == _val_state) &
                    (all_results["_channel"] == _val_ch)
                ]
                if _live_rows.empty:
                    continue
                _live_res = _live_rows.iloc[0]["_result"]
                _mask6 = (
                    (_offline6.get("entity", pd.Series(dtype=str)) == _val_state)
                    & (_offline6.get("model_type", pd.Series(dtype=str)) == "OLS")
                    & (_offline6.get("dummy_family", pd.Series(dtype=str)) == _df_fam)
                )
                _matched6 = _offline6[_mask6]
                _row_data6: dict = {"Iteration": _iter_label}
                _all_match6 = True
                for _m in _metric_cols6:
                    _lv = _live_res.get(_m, np.nan) if isinstance(_live_res, dict) else np.nan
                    _ov = float(_matched6.iloc[0][_m]) if not _matched6.empty and _m in _matched6.columns else np.nan
                    _dv = abs(_lv - _ov) if not (np.isnan(float(_lv)) or np.isnan(float(_ov))) else np.nan
                    _row_data6[f"{_m}_live"] = round(float(_lv), 6) if not np.isnan(float(_lv)) else np.nan
                    _row_data6[f"{_m}_offline"] = round(float(_ov), 6) if not np.isnan(float(_ov)) else np.nan
                    _row_data6[f"{_m}_diff"] = round(float(_dv), 6) if _dv is not None and not np.isnan(_dv) else np.nan
                    if _dv is None or np.isnan(_dv) or _dv >= 0.01:
                        _all_match6 = False
                if _all_match6:
                    _match_count6 += 1
                _cmp_rows6.append(_row_data6)
            if _cmp_rows6:
                st.success(f"✅ {_match_count6}/{len(_cmp_rows6)} baseline(s) matched.") if _match_count6 == len(_cmp_rows6) else st.warning(f"⚠ {len(_cmp_rows6) - _match_count6} differ.")
                _cmp_df6 = pd.DataFrame(_cmp_rows6)
                for _m in _metric_cols6:
                    with st.expander(f"**{_m}**", expanded=(_m == "R2")):
                        _sub6 = _cmp_df6[["Iteration", f"{_m}_live", f"{_m}_offline", f"{_m}_diff"]].copy()
                        _sub6.columns = ["Iteration", "Live", "Offline", "Diff"]
                        def _cd6(_v):
                            if _v is None or (isinstance(_v, float) and np.isnan(_v)):
                                return "color: gray"
                            return "color: green" if abs(_v) < 0.001 else "color: orange" if abs(_v) < 0.01 else "color: red"
                        st.dataframe(_sub6.style.applymap(_cd6, subset=["Diff"]), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.dataframe(pd.DataFrame([
            {"Iteration": c["label"], "Status": "No offline baseline"}
            for c in V6_ITERATIONS if c["num"] not in {1, 2}
        ]), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main — render all tabs
# ---------------------------------------------------------------------------

st.title("Marketing Analytics and Modeling")
st.caption("Channel-specific analysis and regression for DIGITAL and PHYSICAL applications.")

# =============================================================================
# Version 7 — Metric Calculator (step-by-step breakdown)
# =============================================================================
def render_tab_mmm_v7() -> None:
    render_version_intro(
        "Version 7 — Metric Calculator",
        [
            "Uses the updated NON-DM workflow, where Direct Mail applications are removed using `DM_Negative_Differences.xlsx`.",
            "Pick a State, Channel, and Iteration to see exactly how each metric is computed on the NON_DM_APPLICATIONS target.",
            "Shows week-by-week Actual vs Predicted, residuals, and the formula behind MAPE, R², RMSE, OOS RMSE, and Coefficients.",
        ],
        note="Uses the same OLS iteration logic as V6, but with the updated NON_DM_APPLICATIONS source. Train: 2024–2025. OOS test: first 8 weeks of 2026.",
    )

    from utils import MARKETING_SPEND_PATH as _msp7
    from data_processing import (
        DM_DIFF_PATH as _dm_diff7,
        V6_ITERATIONS as _iters7,
        fit_v6_iteration as _fit_v7,
    )

    missing = [n for p, n in [(_msp7, _msp7.name), (_dm_diff7, _dm_diff7.name)] if not p.exists()]
    if missing:
        st.error(f"Missing V7 input files: {', '.join(missing)}")
        return

    c1, c2, c3, c4 = st.columns([2, 1, 3, 1])
    with c1:
        v7_state = st.selectbox("State", _V6_FIXED_STATES, key="v7_state")
    with c2:
        v7_ch = st.selectbox("Channel", ["DIGITAL", "PHYSICAL"], key="v7_ch")
    with c3:
        _iter_labels = [f"{cfg['num']}. {cfg['label']}" for cfg in _iters7]
        v7_iter_label = st.selectbox("Iteration", _iter_labels, key="v7_iter")
        v7_cfg = _iters7[_iter_labels.index(v7_iter_label)]
    with c4:
        st.write("")
        v7_run = st.button("▶ Calculate", key="v7_run")

    if v7_run:
        try:
            _frame = cached_build_v7_modeling_frame(v7_ch)
        except Exception as exc:
            st.error(f"Could not build the V7 NON-DM modeling frame: {exc}")
            return
        _edf = _frame[_frame["STATE_CD"] == v7_state].copy()
        _edf = _edf.sort_values(["ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)
        _res = _call_fit_v6_compat(_fit_v7, _edf, v7_ch, v7_cfg)
        if _res is None:
            st.warning("Model could not be fitted for this selection.")
            return
        st.session_state["v7_result"] = _res
        st.session_state["v7_label"] = f"Iter {v7_cfg['num']} | {v7_state} | {v7_ch}"
        st.session_state["v7_target_note"] = "Target = NON_DM_APPLICATIONS built from DM_Negative_Differences.xlsx"

    _res = st.session_state.get("v7_result")
    if _res is None:
        st.info("Select a State, Channel, and Iteration then click **▶ Calculate**.")
        return

    _label = st.session_state.get("v7_label", "")
    _target_note = st.session_state.get("v7_target_note", "")
    st.markdown(f"### Results: {_label}")
    if _target_note:
        st.caption(_target_note)

    # ------------------------------------------------------------------ #
    # Summary metrics row
    # ------------------------------------------------------------------ #
    y_tr = _res["y_train_actual"]
    yh_tr = _res["y_train_pred"]
    y_te = _res["y_test_actual"]
    yh_te = _res["y_test_pred"]
    oos_rmse = float(np.sqrt(np.mean((y_te - yh_te) ** 2))) if len(y_te) > 0 else np.nan

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("MAPE (train)", f"{_res['MAPE']:.2f}%")
    mc2.metric("R² (train)", f"{_res['R2']:.4f}")
    mc3.metric("RMSE (train)", f"{_res['RMSE']:.2f}")
    mc4.metric("OOS RMSE (test)", f"{oos_rmse:.2f}" if not np.isnan(oos_rmse) else "N/A")

    st.divider()

    calc_tabs = st.tabs([
        "📐 MAPE",
        "📐 R²",
        "📐 RMSE",
        "📐 OOS RMSE",
        "📐 Coefficients",
    ])

    # ---- MAPE --------------------------------------------------------- #
    with calc_tabs[0]:
        st.markdown("""
**Formula:**
> MAPE = mean( |Actual − Predicted| / |Actual| ) × 100
>
> Only weeks where Actual ≠ 0 are included.
""")
        _tr_periods = _res["train_periods"]
        _mape_rows = []
        for i, (yr, wk) in enumerate(_tr_periods):
            act, pred = float(y_tr[i]), float(yh_tr[i])
            err = abs(act - pred)
            pct = (err / abs(act) * 100) if act != 0 else None
            _mape_rows.append({
                "Year": int(yr), "Week": int(wk),
                "Actual": round(act, 2), "Predicted": round(pred, 2),
                "| Actual − Pred |": round(err, 2),
                "% Error ( ÷ Actual × 100)": round(pct, 2) if pct is not None else "excluded (Actual=0)",
            })
        _mape_df = pd.DataFrame(_mape_rows)
        st.dataframe(_mape_df, hide_index=True, use_container_width=True)
        _valid = [r["% Error ( ÷ Actual × 100)"] for r in _mape_rows if isinstance(r["% Error ( ÷ Actual × 100)"], float)]
        st.info(f"Sum of % errors = **{sum(_valid):.2f}** ÷ **{len(_valid)} weeks** = **MAPE {np.mean(_valid):.2f}%**")

    # ---- R² ----------------------------------------------------------- #
    with calc_tabs[1]:
        st.markdown("""
**Formula:**
> R² = 1 − SS_res / SS_tot
>
> SS_res = Σ (Actual − Predicted)²  ← how much the model misses
> SS_tot = Σ (Actual − Mean(Actual))²  ← total variability in the data
>
> R² = 1 means perfect fit. R² = 0 means the model is no better than just predicting the mean.
""")
        _mean_y = float(np.mean(y_tr))
        _ss_res = float(np.sum((y_tr - yh_tr) ** 2))
        _ss_tot = float(np.sum((y_tr - _mean_y) ** 2))
        _r2_calc = 1 - _ss_res / _ss_tot if _ss_tot != 0 else np.nan
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Mean(Actual)", f"{_mean_y:.2f}")
        rc2.metric("SS_res", f"{_ss_res:,.2f}")
        rc3.metric("SS_tot", f"{_ss_tot:,.2f}")
        st.info(f"R² = 1 − ({_ss_res:,.2f} ÷ {_ss_tot:,.2f}) = **{_r2_calc:.6f}**")
        _r2_rows = []
        for i, (yr, wk) in enumerate(_tr_periods):
            act, pred = float(y_tr[i]), float(yh_tr[i])
            _r2_rows.append({
                "Year": int(yr), "Week": int(wk),
                "Actual": round(act, 2), "Predicted": round(pred, 2),
                "(Actual − Pred)²": round((act - pred) ** 2, 2),
                "(Actual − Mean)²": round((act - _mean_y) ** 2, 2),
            })
        st.dataframe(pd.DataFrame(_r2_rows), hide_index=True, use_container_width=True)

    # ---- RMSE --------------------------------------------------------- #
    with calc_tabs[2]:
        st.markdown("""
**Formula:**
> RMSE = √ mean( (Actual − Predicted)² )
>
> It's in the same units as Applications — lower is better.
""")
        _sq_errs = (y_tr - yh_tr) ** 2
        _mse = float(np.mean(_sq_errs))
        _rmse_calc = float(np.sqrt(_mse))
        rc1, rc2 = st.columns(2)
        rc1.metric("Mean Squared Error", f"{_mse:,.4f}")
        rc2.metric("RMSE = √MSE", f"{_rmse_calc:.4f}")
        _rmse_rows = []
        for i, (yr, wk) in enumerate(_tr_periods):
            act, pred = float(y_tr[i]), float(yh_tr[i])
            sq_err = (act - pred) ** 2
            _rmse_rows.append({
                "Year": int(yr), "Week": int(wk),
                "Actual": round(act, 2), "Predicted": round(pred, 2),
                "Error": round(act - pred, 2),
                "Squared Error": round(sq_err, 2),
            })
        st.dataframe(pd.DataFrame(_rmse_rows), hide_index=True, use_container_width=True)
        st.info(f"Sum of squared errors = {sum(r['Squared Error'] for r in _rmse_rows):,.2f} ÷ {len(_rmse_rows)} weeks → MSE = {_mse:,.4f} → RMSE = **{_rmse_calc:.4f}**")

    # ---- OOS RMSE ----------------------------------------------------- #
    with calc_tabs[3]:
        st.markdown("""
**Formula:** Same as RMSE but on the **test set** (first 8 weeks of 2026).
> OOS RMSE = √ mean( (Actual − Predicted)² )  — computed on weeks the model never trained on.
>
> This is the most important metric: it shows whether the model generalises beyond the training data.
""")
        if len(y_te) == 0:
            st.warning("No test data available for this selection.")
        else:
            _te_periods = _res["test_periods"]
            _oos_rows = []
            for i, (yr, wk) in enumerate(_te_periods):
                act, pred = float(y_te[i]), float(yh_te[i])
                sq_err = (act - pred) ** 2
                _oos_rows.append({
                    "Year": int(yr), "Week": int(wk),
                    "Actual": round(act, 2), "Predicted": round(pred, 2),
                    "Error": round(act - pred, 2),
                    "Squared Error": round(sq_err, 2),
                })
            _oos_mse = float(np.mean([(r["Squared Error"]) for r in _oos_rows]))
            _oos_rmse_calc = float(np.sqrt(_oos_mse))
            st.dataframe(pd.DataFrame(_oos_rows), hide_index=True, use_container_width=True)
            st.info(f"Sum of squared errors = {sum(r['Squared Error'] for r in _oos_rows):,.2f} ÷ {len(_oos_rows)} test weeks → MSE = {_oos_mse:,.4f} → OOS RMSE = **{_oos_rmse_calc:.4f}**")

    # ---- Coefficients ------------------------------------------------- #
    with calc_tabs[4]:
        st.markdown("""
**How OLS coefficients work:**
> The model fits:  Predicted = β₁·X₁ + β₂·X₂ + … (no intercept)
>
> Tactic columns (Prescreen, DSP, etc.) are **MinMax scaled** to [0, 1] before fitting,
> so coefficients are on the same scale and comparable across tactics.
> Seasonal dummy columns are unscaled (already 0 or 1).
>
> A **positive coefficient** means more spend → more applications.
> A **negative coefficient** is a warning sign (multicollinearity or overfitting).
""")
        _coef_rows = []
        for feat, coef in zip(_res["features"], _res["coefs"]):
            _is_tactic = feat in _res.get("tactic_cols", [])
            _coef_rows.append({
                "Feature": feat,
                "Type": "Tactic (MinMax scaled)" if _is_tactic else "Seasonal dummy",
                "Coefficient": round(float(coef), 6),
                "Direction": "Positive ✓" if coef > 0 else "Negative ⚠",
            })
        st.dataframe(pd.DataFrame(_coef_rows), hide_index=True, use_container_width=True)
        _tac_coefs = [(r["Feature"], r["Coefficient"]) for r in _coef_rows if r["Type"].startswith("Tactic")]
        if _tac_coefs:
            st.markdown("**Tactic coefficient summary (scaled values):**")
            for f, c in sorted(_tac_coefs, key=lambda x: -abs(x[1])):
                bar = "█" * min(int(abs(c) / max(abs(cc) for _, cc in _tac_coefs) * 20), 20)
                st.text(f"  {f:<20} {c:+.4f}  {bar}")


def render_tab_mmm_v8() -> None:
    render_version_intro(
        "Version 8 — With DM vs Without DM Comparison",
        [
            "Confirms the V6/V7 logic: **V6 already excludes DM applications** because it models `NON_DM_APPLICATIONS`.",
            "Runs the same 16 OLS iterations twice for each AL, CA, DE, FL state and for both DIGITAL and PHYSICAL channels.",
            "Comparison 1 uses `APPLICATIONS` (with DM included). Comparison 2 uses `NON_DM_APPLICATIONS` (DM removed using `DM_Negative_Differences.xlsx`).",
            "Shows how `MAPE`, `R²`, and `RMSE` change by iteration when DM applications are included vs excluded.",
            "Highlights the Direct Mail impact itself with `APPLICATIONS`, `DM_APPLICATIONS`, and `NON_DM_APPLICATIONS` over time.",
        ],
        note="Train: 2024 + 2025. Test: first 8 weeks of 2026. Same V6 iteration settings, only the target changes.",
    )

    from utils import (
        MARKETING_SPEND_PATH as _msp8,
        ORIGINATIONS_V5_PATH as _op8,
    )
    from data_processing import DM_DIFF_PATH as _dm_diff8, fit_v6_iteration as _fit_v8

    missing = [n for p, n in [(_msp8, _msp8.name), (_op8, _op8.name), (_dm_diff8, _dm_diff8.name)] if not p.exists()]
    if missing:
        st.error(f"Missing V8 input files: {', '.join(missing)}")
        return

    st.caption("`With DM` uses total APPLICATIONS. `Without DM` uses NON_DM_APPLICATIONS = max(0, APPLICATIONS − DM_APPLICATIONS).")

    v8_run = st.button("▶ Run DM Comparison", key="v8_run")
    if v8_run:
        _target_specs = [
            {"target_label": "With DM", "target_col": "APPLICATIONS"},
            {"target_label": "Without DM", "target_col": "NON_DM_APPLICATIONS"},
        ]
        all_rows = []
        with st.spinner("Running V8 comparison across all states, channels, and iterations…"):
            for _ch in ["DIGITAL", "PHYSICAL"]:
                try:
                    _frame = cached_build_v7_modeling_frame(_ch)
                except Exception as exc:
                    st.warning(f"Could not build V8 frame for {_ch}: {exc}")
                    continue

                for _state in _V6_FIXED_STATES:
                    _edf = _frame[_frame["STATE_CD"] == _state].copy()
                    _edf = _edf.sort_values(["ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)
                    for _cfg in V6_ITERATIONS:
                        for _target in _target_specs:
                            _res = _call_fit_v6_compat(
                                _fit_v8,
                                _edf,
                                _ch,
                                _cfg,
                                extra_kwargs={"target_col": _target["target_col"]},
                            )
                            if _res is None:
                                continue
                            all_rows.append({
                                "iter_num": _cfg["num"],
                                "Iteration": _cfg["label"],
                                "_state": _state,
                                "_channel": _ch,
                                "target_label": _target["target_label"],
                                "target_col": _target["target_col"],
                                "MAPE": round(float(_res["MAPE"]), 2),
                                "R. Sq": round(float(_res["R2"]), 4),
                                "RMSE": round(float(_res["RMSE"]), 2),
                                "OOS RMSE": round(float(np.sqrt(np.mean((_res["y_test_actual"] - _res["y_test_pred"]) ** 2))), 2)
                                if len(_res["y_test_actual"]) > 0 else np.nan,
                                "_result": _res,
                            })
        st.session_state["v8_all"] = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

    all_results: pd.DataFrame | None = st.session_state.get("v8_all")
    if all_results is None or all_results.empty:
        st.info("Click **▶ Run DM Comparison** to compare each iteration with and without DM applications.")
        return

    _overview = (
        all_results.pivot_table(
            index=["_state", "_channel"],
            columns="target_label",
            values=["MAPE", "R. Sq", "RMSE"],
            aggfunc="mean",
        )
        .sort_index()
    )
    _overview.columns = [f"{metric} | {target}" for metric, target in _overview.columns]
    overview_df = _overview.reset_index().rename(columns={"_state": "State", "_channel": "Channel"})
    overview_df["Delta MAPE (Without - With)"] = overview_df["MAPE | Without DM"] - overview_df["MAPE | With DM"]
    overview_df["Delta R² (Without - With)"] = overview_df["R. Sq | Without DM"] - overview_df["R. Sq | With DM"]
    overview_df["Delta RMSE (Without - With)"] = overview_df["RMSE | Without DM"] - overview_df["RMSE | With DM"]

    st.markdown("**State and Channel Overview**")
    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "MAPE | With DM": st.column_config.NumberColumn("Avg MAPE | With DM (%)", format="%.2f"),
            "MAPE | Without DM": st.column_config.NumberColumn("Avg MAPE | Without DM (%)", format="%.2f"),
            "Delta MAPE (Without - With)": st.column_config.NumberColumn("Delta MAPE", format="%.2f"),
            "R. Sq | With DM": st.column_config.NumberColumn("Avg R² | With DM", format="%.4f"),
            "R. Sq | Without DM": st.column_config.NumberColumn("Avg R² | Without DM", format="%.4f"),
            "Delta R² (Without - With)": st.column_config.NumberColumn("Delta R²", format="%.4f"),
            "RMSE | With DM": st.column_config.NumberColumn("Avg RMSE | With DM", format="%.2f"),
            "RMSE | Without DM": st.column_config.NumberColumn("Avg RMSE | Without DM", format="%.2f"),
            "Delta RMSE (Without - With)": st.column_config.NumberColumn("Delta RMSE", format="%.2f"),
        },
    )

    overview_heat = overview_df.copy()
    overview_heat["combo"] = overview_heat["State"] + " | " + overview_heat["Channel"]
    fig_overview_heat = go.Figure(data=go.Heatmap(
        z=overview_heat[["Delta MAPE (Without - With)", "Delta R² (Without - With)", "Delta RMSE (Without - With)"]].values,
        x=["Delta MAPE", "Delta R²", "Delta RMSE"],
        y=overview_heat["combo"],
        colorscale="RdYlGn_r",
        hovertemplate="State|Channel: %{y}<br>%{x}: %{z:.4f}<extra></extra>",
    ))
    fig_overview_heat.update_layout(
        title="Average Metric Change After Removing DM Applications",
        height=max(320, 36 * len(overview_heat)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig_overview_heat, use_container_width=True, key="v8_overview_heat")
    st.caption("For this overview: lower Delta MAPE / Delta RMSE is better, while higher Delta R² is better.")

    fc1, fc2 = st.columns(2)
    with fc1:
        v8_state = st.selectbox("State", _V6_FIXED_STATES, key="v8_state")
    with fc2:
        v8_channel = st.selectbox("Channel", ["DIGITAL", "PHYSICAL"], key="v8_channel")

    selected = (
        all_results[
            (all_results["_state"] == v8_state) &
            (all_results["_channel"] == v8_channel)
        ]
        .copy()
        .sort_values(["iter_num", "target_label"])
    )
    if selected.empty:
        st.warning("No V8 comparison results are available for this state/channel.")
        return

    impact_frame = cached_build_v7_modeling_frame(v8_channel)
    impact_frame = (
        impact_frame[impact_frame["STATE_CD"] == v8_state]
        .copy()
        .sort_values(["ISO_YEAR", "ISO_WEEK"])
        .reset_index(drop=True)
    )
    impact_frame["PERIOD"] = impact_frame["ISO_YEAR"].astype(int).astype(str) + "-W" + impact_frame["ISO_WEEK"].astype(int).astype(str).str.zfill(2)
    impact_frame["DM_REMOVED"] = (impact_frame["APPLICATIONS"] - impact_frame["NON_DM_APPLICATIONS"]).clip(lower=0)

    total_apps = float(impact_frame["APPLICATIONS"].sum())
    total_dm = float(impact_frame["DM_REMOVED"].sum())
    total_non_dm = float(impact_frame["NON_DM_APPLICATIONS"].sum())
    dm_share = (total_dm / total_apps * 100) if total_apps > 0 else np.nan

    st.markdown("**DM Impact Snapshot**")
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Applications", f"{total_apps:,.0f}")
    dc2.metric("DM Applications Removed", f"{total_dm:,.0f}")
    dc3.metric("Non-DM Applications", f"{total_non_dm:,.0f}")
    dc4.metric("DM Share", f"{dm_share:.2f}%" if not np.isnan(dm_share) else "N/A")
    st.caption("`DM Applications Removed` is the negative difference highlighted from `DM_Negative_Differences.xlsx` after clipping at zero.")

    fig_impact = go.Figure()
    fig_impact.add_trace(go.Bar(
        x=impact_frame["PERIOD"],
        y=impact_frame["DM_REMOVED"],
        name="DM Applications Removed",
        marker_color="#E45756",
        opacity=0.45,
    ))
    fig_impact.add_trace(go.Scatter(
        x=impact_frame["PERIOD"],
        y=impact_frame["APPLICATIONS"],
        mode="lines+markers",
        name="Applications (With DM)",
        line=dict(color="#4C78A8", width=2),
    ))
    fig_impact.add_trace(go.Scatter(
        x=impact_frame["PERIOD"],
        y=impact_frame["NON_DM_APPLICATIONS"],
        mode="lines+markers",
        name="Non-DM Applications",
        line=dict(color="#54A24B", width=2),
    ))
    fig_impact.update_layout(
        title=f"DM Difference Over Time — {v8_state} {v8_channel}",
        xaxis_title="Period",
        yaxis_title="Applications",
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_impact, use_container_width=True, key="v8_dm_impact_chart")

    st.divider()
    st.markdown("**Iteration Comparison**")

    plot_df = selected.copy()
    plot_df["iter_num_label"] = plot_df["iter_num"].astype(int).astype(str)
    color_map = {"With DM": "#4C78A8", "Without DM": "#F58518"}

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        fig_v8_mape = px.line(
            plot_df,
            x="iter_num_label",
            y="MAPE",
            color="target_label",
            markers=True,
            color_discrete_map=color_map,
            labels={"iter_num_label": "Iteration", "MAPE": "MAPE (%)", "target_label": "Target"},
            title="MAPE by Iteration",
        )
        fig_v8_mape.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_v8_mape, use_container_width=True, key="v8_mape_chart")

    with gc2:
        fig_v8_r2 = px.line(
            plot_df,
            x="iter_num_label",
            y="R. Sq",
            color="target_label",
            markers=True,
            color_discrete_map=color_map,
            labels={"iter_num_label": "Iteration", "R. Sq": "R²", "target_label": "Target"},
            title="R² by Iteration",
        )
        fig_v8_r2.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_v8_r2, use_container_width=True, key="v8_r2_chart")

    with gc3:
        fig_v8_rmse = px.line(
            plot_df,
            x="iter_num_label",
            y="RMSE",
            color="target_label",
            markers=True,
            color_discrete_map=color_map,
            labels={"iter_num_label": "Iteration", "RMSE": "RMSE", "target_label": "Target"},
            title="RMSE by Iteration",
        )
        fig_v8_rmse.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_v8_rmse, use_container_width=True, key="v8_rmse_chart")

    _best_with_dm = plot_df[plot_df["target_label"] == "With DM"].sort_values(["MAPE", "R. Sq"], ascending=[True, False]).iloc[0]
    _best_without_dm = plot_df[plot_df["target_label"] == "Without DM"].sort_values(["MAPE", "R. Sq"], ascending=[True, False]).iloc[0]
    bc1, bc2 = st.columns(2)
    bc1.metric("Best With-DM Iteration", str(int(_best_with_dm["iter_num"])), f"MAPE {_best_with_dm['MAPE']:.2f}% | R² {_best_with_dm['R. Sq']:.4f}")
    bc2.metric("Best Without-DM Iteration", str(int(_best_without_dm["iter_num"])), f"MAPE {_best_without_dm['MAPE']:.2f}% | R² {_best_without_dm['R. Sq']:.4f}")

    pivot = (
        selected.pivot_table(
            index=["iter_num", "Iteration"],
            columns="target_label",
            values=["MAPE", "R. Sq", "RMSE"],
            aggfunc="first",
        )
        .sort_index()
    )
    pivot.columns = [f"{metric} | {target}" for metric, target in pivot.columns]
    compare_df = pivot.reset_index()
    compare_df["Delta MAPE (Without - With)"] = compare_df["MAPE | Without DM"] - compare_df["MAPE | With DM"]
    compare_df["Delta R² (Without - With)"] = compare_df["R. Sq | Without DM"] - compare_df["R. Sq | With DM"]
    compare_df["Delta RMSE (Without - With)"] = compare_df["RMSE | Without DM"] - compare_df["RMSE | With DM"]
    compare_df.rename(columns={"iter_num": "Iteration #"}, inplace=True)

    st.markdown("**With vs Without DM by Iteration**")
    st.dataframe(
        compare_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Iteration #": st.column_config.NumberColumn("Iteration #", format="%d"),
            "Iteration": st.column_config.TextColumn("Iteration", width="large"),
            "MAPE | With DM": st.column_config.NumberColumn("MAPE | With DM (%)", format="%.2f"),
            "MAPE | Without DM": st.column_config.NumberColumn("MAPE | Without DM (%)", format="%.2f"),
            "Delta MAPE (Without - With)": st.column_config.NumberColumn("Delta MAPE", format="%.2f"),
            "R. Sq | With DM": st.column_config.NumberColumn("R² | With DM", format="%.4f"),
            "R. Sq | Without DM": st.column_config.NumberColumn("R² | Without DM", format="%.4f"),
            "Delta R² (Without - With)": st.column_config.NumberColumn("Delta R²", format="%.4f"),
            "RMSE | With DM": st.column_config.NumberColumn("RMSE | With DM", format="%.2f"),
            "RMSE | Without DM": st.column_config.NumberColumn("RMSE | Without DM", format="%.2f"),
            "Delta RMSE (Without - With)": st.column_config.NumberColumn("Delta RMSE", format="%.2f"),
        },
    )
    st.caption(
        "Negative Delta MAPE / Delta RMSE means the NON-DM model improved after removing DM applications. "
        "Positive Delta R² means the NON-DM model explained more variance."
    )


# =============================================================================
# Tab layout
# =============================================================================
tabs = st.tabs([
    "📊 Marketing Spend EDA",
    "📈 Originations EDA",
    "🔬 V1",
    "🧪 V2",
    "🧩 V3",
    "🧱 V4",
    "🔬 V5",
    "📐 V6",
    "🧮 V7",
    "⚖️ V8",
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

with tabs[6]:
    render_tab_mmm_v5()

with tabs[7]:
    render_tab_mmm_v6()

with tabs[8]:
    render_tab_mmm_v7()

with tabs[9]:
    render_tab_mmm_v8()
