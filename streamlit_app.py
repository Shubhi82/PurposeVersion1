from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processing import (
    build_application_series,
    build_channel_comparison_table,
    build_product_spend_series,
    build_tactic_time_series,
    load_marketing_spend_data,
    load_marketing_spend_raw,
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
)
from modeling import fit_channel_models
from utils import (
    CHANNELS,
    DEFAULT_DATA_PATH,
    MARKETING_SPEND_PATH,
    ORIGINATIONS_PATH,
    OUTCOME_COLUMNS,
    PRODUCT_ALL_LABEL,
    TACTIC_COLUMNS,
    TIME_GRAINS,
    format_metric,
    get_available_products,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Marketing Analytics and Modeling", layout="wide")

PALETTE = px.colors.qualitative.Set2

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def chart_with_table(fig, table_df: pd.DataFrame, table_label: str = "View data") -> None:
    """Render a plotly figure then show its underlying data in a collapsible expander."""
    st.plotly_chart(fig, use_container_width=True)
    with st.expander(f"📋 {table_label}"):
        st.dataframe(table_df, use_container_width=True, hide_index=True)

def apply_bottom_legend(fig: go.Figure) -> go.Figure:
    """Move legend to bottom of the chart and increase bottom margin."""
    fig.update_layout(
        margin=dict(l=10, r=10, t=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )
    return fig

def bar_h(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    fig = px.bar(
        df, x=x, y=y, orientation="h", title=title,
        color=color, color_discrete_sequence=PALETTE,
        text_auto=".3s",
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

def bar_v(df: pd.DataFrame, x: str, y: str | list, title: str, barmode: str = "group") -> go.Figure:
    if isinstance(y, list):
        fig = px.bar(
            df, x=x, y=y, title=title, barmode=barmode,
            color_discrete_sequence=PALETTE, text_auto=".3s",
        )
    else:
        fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=PALETTE, text_auto=".3s")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def line_chart(df: pd.DataFrame, x: str, y: str | list, title: str) -> go.Figure:
    fig = px.line(df, x=x, y=y, title=title, color_discrete_sequence=PALETTE, markers=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def heatmap_fig(df: pd.DataFrame, title: str) -> go.Figure:
    idx_col = df.columns[0]
    mat = df.set_index(idx_col)
    fig = go.Figure(data=go.Heatmap(
        z=mat.values,
        x=mat.columns.tolist(),
        y=mat.index.tolist(),
        colorscale="Blues",
        text=[[f"${v:,.0f}" for v in row] for row in mat.values],
        texttemplate="%{text}",
        hovertemplate="%{y} | %{x}: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ---------------------------------------------------------------------------
# Cached auto-loaders — read from files bundled with the repo
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading marketing spend data…")
def auto_load_marketing_spend() -> pd.DataFrame:
    return load_marketing_spend_data(MARKETING_SPEND_PATH, fallback_source=DEFAULT_DATA_PATH)

@st.cache_data(show_spinner="Loading originations data…")
def auto_load_originations() -> pd.DataFrame:
    return load_originations_data(ORIGINATIONS_PATH, fallback_source=DEFAULT_DATA_PATH)

@st.cache_data(show_spinner="Loading modeling workbook…")
def auto_load_modeling(path: str) -> pd.DataFrame:
    return load_modeling_data(Path(path))

@st.cache_data(show_spinner=False)
def cached_load_data(source: bytes) -> pd.DataFrame:
    return load_modeling_data(source)

# ---------------------------------------------------------------------------
# Tab 3 — Marketing Analysis
# ---------------------------------------------------------------------------

def render_channel_chart_expander(channel: str, chart_data: pd.DataFrame, metric_label: str) -> None:
    st.markdown(f"#### {channel}")
    if chart_data.empty:
        st.info(f"No {channel.lower()} {metric_label.lower()} data available for the current filter.")
        return
    df_plot = chart_data.reset_index()
    period_col = df_plot.columns[0]
    value_cols = [c for c in df_plot.columns if c != period_col]
    fig = px.line(df_plot, x=period_col, y=value_cols, markers=True,
                  color_discrete_sequence=PALETTE)
    fig = apply_bottom_legend(fig)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 View data"):
        st.dataframe(df_plot, use_container_width=True, hide_index=True)

def render_product_bar_chart(channel: str, chart_data: pd.DataFrame) -> None:
    st.markdown(f"#### {channel}")
    if chart_data.empty:
        st.info(f"No {channel.lower()} product spend data available.")
        return
    df_plot = chart_data.reset_index()
    period_col = df_plot.columns[0]
    value_cols = [c for c in df_plot.columns if c != period_col]
    fig = px.bar(
        df_plot, x=period_col, y=value_cols, barmode="stack",
        color_discrete_sequence=PALETTE,
        labels={"value": "Spend ($)", "variable": "Tactic"},
    )
    fig = apply_bottom_legend(fig)
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 View data"):
        st.dataframe(df_plot, use_container_width=True, hide_index=True)

def render_model_panel(channel: str, result) -> None:
    st.markdown(f"#### {channel} model")
    mape_display = "N/A" if pd.isna(result.mape) else f"{format_metric(result.mape, 2)}%"
    st.metric("MAPE", mape_display)

    st.caption(
        f"**Y-axis (dependent):** `Applications_{channel}` · "
        f"**X-axis (predictors):** channel-specific spend tactics + time dummies "
        f"({', '.join(result.periods_used[:4])}{' ...' if len(result.periods_used) > 4 else ''})."
    )

    chart_frame = result.fitted_frame.set_index("period_label")[
        ["APPLICATIONS", "Predicted_Applications"]
    ]
    fig = px.line(
        chart_frame.reset_index(), x="period_label",
        y=["APPLICATIONS", "Predicted_Applications"],
        markers=True, color_discrete_sequence=PALETTE,
        labels={"value": "Applications", "variable": "Series", "period_label": "Period"},
    )
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
