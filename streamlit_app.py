from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processing import (
    aggregate_for_analysis,
    build_application_series,
    build_channel_comparison_table,
    build_product_spend_series,
    build_tactic_time_series,
    load_marketing_spend_raw,
    load_modeling_data,
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


def bar_h(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    fig = px.bar(
        df, x=x, y=y, orientation="h", title=title,
        color=color, color_discrete_sequence=PALETTE,
        text_auto=".3s",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def bar_v(df: pd.DataFrame, x: str, y: str | list, title: str, barmode: str = "group") -> go.Figure:
    if isinstance(y, list):
        fig = px.bar(df, x=x, y=y, title=title, barmode=barmode,
                     color_discrete_sequence=PALETTE, text_auto=".3s")
    else:
        fig = px.bar(df, x=x, y=y, title=title,
                     color_discrete_sequence=PALETTE, text_auto=".3s")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def line_chart(df: pd.DataFrame, x: str, y: str | list, title: str) -> go.Figure:
    if isinstance(y, list):
        fig = px.line(df, x=x, y=y, title=title, color_discrete_sequence=PALETTE, markers=True)
    else:
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
# Tab 1 — EDA: Marketing Spend
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_ms(data: bytes) -> pd.DataFrame:
    return load_marketing_spend_raw(data)


def render_tab_marketing_spend() -> None:
    st.header("Exploratory Data Analysis — Marketing Spend")
    uploaded = st.file_uploader(
        "Upload Marketing_Spend_Data.csv", type=["csv"], key="ms_upload"
    )
    if uploaded is None:
        st.info("Upload the Marketing_Spend_Data.csv file to begin EDA.")
        return

    df = cached_ms(uploaded.getvalue())
    st.caption(f"Loaded **{len(df):,}** rows across **{df['DETAIL_TACTIC'].nunique()}** tactics, "
               f"**{df['STATE_CD'].nunique() if 'STATE_CD' in df.columns else '—'}** states.")

    # --- 1. Spend by Tactic ---
    st.subheader("1 · Spend by Tactic")
    tactic_df = ms_spend_by_tactic(df)
    fig1 = bar_h(tactic_df, x="Total Spend ($)", y="Tactic", title="Total Spend by Tactic")
    chart_with_table(fig1, tactic_df, "Spend by Tactic")

    # --- 2. Spend by Channel ---
    st.subheader("2 · Spend by Channel")
    channel_df = ms_spend_by_channel(df)
    fig2 = bar_v(channel_df, x="Channel", y="Total Spend ($)", title="Total Spend by Channel")
    chart_with_table(fig2, channel_df, "Spend by Channel")

    # --- 3. Spend by State ---
    st.subheader("3 · Spend by State")
    state_df = ms_spend_by_state(df)
    fig3 = bar_h(state_df, x="Total Spend ($)", y="State", title="Total Spend by State")
    chart_with_table(fig3, state_df, "Spend by State")

    # --- 4. Spend by Product ---
    st.subheader("4 · Spend by Product")
    prod_df = ms_spend_by_product(df)
    fig4 = bar_h(prod_df, x="Total Spend ($)", y="Product", title="Total Spend by Product")
    chart_with_table(fig4, prod_df, "Spend by Product")

    # --- 5. Spend Over Time ---
    st.subheader("5 · Weekly Spend Over Time")
    time_df = ms_spend_over_time(df)
    if not time_df.empty:
        fig5 = line_chart(time_df, x="period", y="TOTAL_COST", title="Weekly Total Spend Over Time")
        chart_with_table(fig5, time_df[["period", "TOTAL_COST"]].rename(columns={"TOTAL_COST": "Spend ($)"}),
                         "Weekly Spend")
    else:
        st.info("No time-series data available.")

    # --- 6. Tactic × Channel Heatmap ---
    st.subheader("6 · Tactic × Channel Spend Heatmap")
    matrix_df = ms_tactic_channel_matrix(df)
    if not matrix_df.empty:
        fig6 = heatmap_fig(matrix_df, "Spend Heatmap — Tactic × Channel")
        chart_with_table(fig6, matrix_df, "Tactic × Channel Matrix")

    # --- 7. Spend by State × Tactic (stacked bar) ---
    st.subheader("7 · Spend by State & Tactic")
    st_tactic = ms_spend_by_state_tactic(df)
    if not st_tactic.empty:
        fig7 = px.bar(
            st_tactic, x="STATE_CD", y="TOTAL_COST", color="DETAIL_TACTIC",
            title="Spend by State and Tactic", barmode="stack",
            color_discrete_sequence=PALETTE, text_auto=".3s",
            labels={"STATE_CD": "State", "TOTAL_COST": "Spend ($)", "DETAIL_TACTIC": "Tactic"},
        )
        fig7.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        table7 = st_tactic.rename(columns={"STATE_CD": "State", "DETAIL_TACTIC": "Tactic", "TOTAL_COST": "Spend ($)"})
        chart_with_table(fig7, table7, "State × Tactic Spend")


# ---------------------------------------------------------------------------
# Tab 2 — EDA: Originations
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_od(data: bytes) -> pd.DataFrame:
    return load_originations_raw(data)


def render_tab_originations() -> None:
    st.header("Exploratory Data Analysis — Originations")
    uploaded = st.file_uploader(
        "Upload Originations_Data.xlsx", type=["csv"], key="od_upload"
    )
    if uploaded is None:
        st.info("Upload the Originations_Data.xlsx file to begin EDA.")
        return

    df = cached_od(uploaded.getvalue())
    total_apps = int(df["APPLICATIONS"].sum()) if "APPLICATIONS" in df.columns else 0
    total_appr = int(df["APPROVED"].sum()) if "APPROVED" in df.columns else 0
    total_orig = int(df["ORIGINATIONS"].sum()) if "ORIGINATIONS" in df.columns else 0
    st.caption(
        f"Loaded **{len(df):,}** rows · "
        f"Applications: **{total_apps:,}** · "
        f"Approved: **{total_appr:,}** · "
        f"Funded: **{total_orig:,}**"
    )

    # --- 1. Metrics by State ---
    st.subheader("1 · Applications, Approvals & Funding by State")
    state_df = od_metrics_by_state(df)
    metric_cols = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in state_df.columns]
    fig1 = bar_v(state_df, x="State", y=metric_cols,
                 title="Applications / Approvals / Funding by State", barmode="group")
    chart_with_table(fig1, state_df, "Metrics by State")

    # --- 2. Metrics by Channel ---
    st.subheader("2 · Metrics by Channel")
    ch_df = od_metrics_by_channel(df)
    metric_cols2 = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in ch_df.columns]
    fig2 = bar_v(ch_df, x="Channel", y=metric_cols2,
                 title="Applications / Approvals / Funding by Channel", barmode="group")
    chart_with_table(fig2, ch_df, "Metrics by Channel")

    # --- 3. Metrics by Product ---
    st.subheader("3 · Metrics by Product")
    prod_df = od_metrics_by_product(df)
    metric_cols3 = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in prod_df.columns]
    fig3 = bar_h(prod_df.melt(id_vars=["Product"], value_vars=metric_cols3, var_name="Metric", value_name="Count"),
                 x="Count", y="Product", title="Metrics by Product", color="Metric")
    chart_with_table(fig3, prod_df, "Metrics by Product")

    # --- 4. Weekly Trends ---
    st.subheader("4 · Weekly Trends — Applications, Approvals & Funding")
    time_df = od_metrics_over_time(df)
    if not time_df.empty:
        metric_cols4 = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in time_df.columns]
        fig4 = line_chart(time_df, x="period", y=metric_cols4,
                          title="Weekly Applications, Approvals & Funding Over Time")
        chart_with_table(fig4, time_df[["period"] + metric_cols4], "Weekly Trends")

    # --- 5. Funnel Rates by State ---
    st.subheader("5 · Approval & Funding Rate by State")
    funnel_df = od_funnel_by_state(df)
    rate_cols = [c for c in ["Approval Rate (%)", "Funding Rate (%)"] if c in funnel_df.columns]
    if rate_cols:
        fig5 = bar_v(funnel_df, x="State", y=rate_cols,
                     title="Approval Rate & Funding Rate by State (%)", barmode="group")
        chart_with_table(fig5, funnel_df[["State"] + rate_cols], "Funnel Rates by State")

    # --- 6. Channel × State Breakdown ---
    st.subheader("6 · Channel × State Breakdown")
    cs_df = od_channel_state_matrix(df)
    if not cs_df.empty and "APPLICATIONS" in cs_df.columns:
        fig6 = px.bar(
            cs_df, x="STATE_CD", y="APPLICATIONS", color="CHANNEL_CD",
            barmode="group", title="Applications by State and Channel",
            color_discrete_sequence=PALETTE,
            labels={"STATE_CD": "State", "APPLICATIONS": "Applications", "CHANNEL_CD": "Channel"},
        )
        fig6.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        chart_with_table(fig6, cs_df.rename(columns={"STATE_CD": "State", "CHANNEL_CD": "Channel"}),
                         "Channel × State Data")


# ---------------------------------------------------------------------------
# Tab 3 — Marketing Analysis (existing, upgraded)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_load_data(source: bytes) -> pd.DataFrame:
    return load_modeling_data(source)


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
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 View data"):
        st.dataframe(df_plot, use_container_width=True, hide_index=True)


def render_product_bar_chart(channel: str, chart_data: pd.DataFrame) -> None:
    """Section 2 — stacked bar instead of line chart."""
    st.markdown(f"#### {channel}")
    if chart_data.empty:
        st.info(f"No {channel.lower()} product spend data available.")
        return
    df_plot = chart_data.reset_index()
    period_col = df_plot.columns[0]
    value_cols = [c for c in df_plot.columns if c != period_col]
    fig = px.bar(df_plot, x=period_col, y=value_cols, barmode="stack",
                 color_discrete_sequence=PALETTE,
                 labels={"value": "Spend ($)", "variable": "Tactic"})
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("📋 View data"):
        st.dataframe(df_plot, use_container_width=True, hide_index=True)


def render_model_panel(channel: str, result) -> None:
    st.markdown(f"#### {channel} model")
    mape_display = "N/A" if pd.isna(result.mape) else f"{format_metric(result.mape, 2)}%"
    st.metric("MAPE", mape_display)

    # Y = Applications by Channel | X = Tactics + Week dummies
    st.caption(
        f"**Y-axis (dependent):** `Applications_{channel}` · "
        f"**X-axis (predictors):** channel-specific spend tactics + time dummies "
        f"({', '.join(result.periods_used[:4])}{' ...' if len(result.periods_used) > 4 else ''})."
    )

    # Actual vs Predicted
    chart_frame = result.fitted_frame.set_index("period_label")[["APPLICATIONS", "Predicted_Applications"]]
    fig = px.line(chart_frame.reset_index(), x="period_label",
                  y=["APPLICATIONS", "Predicted_Applications"],
                  markers=True, color_discrete_sequence=PALETTE,
                  labels={"value": "Applications", "variable": "Series", "period_label": "Period"})
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
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

    # Tactic coefficients
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
        coef_fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(coef_fig, use_container_width=True)
        with st.expander("📋 Coefficient table"):
            st.dataframe(tactic_coefs, use_container_width=True, hide_index=True)

    if result.narrative:
        st.markdown("**Interpretation**")
        for sentence in result.narrative:
            st.write(sentence)


def render_tab_marketing_analysis() -> None:
    # --- File upload only (no workbook path shown) ---
    uploaded_file = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx"], key="model_upload")

    col_agg, = st.columns([1])
    with col_agg:
        time_grain = st.selectbox("Aggregation", TIME_GRAINS)

    if uploaded_file is None:
        # Try default path silently
        if DEFAULT_DATA_PATH.exists():
            try:
                data = load_modeling_data(DEFAULT_DATA_PATH)
            except Exception:
                st.info("Upload the consolidated workbook (.xlsx) to begin analysis.")
                return
        else:
            st.info("Upload the consolidated workbook (.xlsx) to begin analysis.")
            return
    else:
        try:
            data = cached_load_data(uploaded_file.getvalue())
        except Exception as exc:
            st.error(f"Unable to load workbook: {exc}")
            return

    state_options = sorted(data["STATE_CD"].dropna().unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        state = st.selectbox("STATE", state_options)
    with c2:
        product = st.selectbox("PRODUCT (optional)", get_available_products(data, state))

    # ------------------------------------------------------------------ #
    # Section 1: Marketing spend consistency
    # ------------------------------------------------------------------ #
    st.subheader("Section 1: Marketing Spend Consistency")
    tactic_series = build_tactic_time_series(
        data, time_grain=time_grain, state=state,
        product=None if product == PRODUCT_ALL_LABEL else product,
    )
    left_col, right_col = st.columns(2)
    with left_col:
        render_channel_chart_expander("DIGITAL", tactic_series["DIGITAL"], "spend")
    with right_col:
        render_channel_chart_expander("PHYSICAL", tactic_series["PHYSICAL"], "spend")

    # ------------------------------------------------------------------ #
    # Section 2: Product-level marketing consistency
    # ------------------------------------------------------------------ #
    st.subheader("Section 2: Product-Level Marketing Consistency")
    if product == PRODUCT_ALL_LABEL:
        st.info("Select a specific product to compare DIGITAL and PHYSICAL spend over time.")
    else:
        product_series = build_product_spend_series(
            data, time_grain=time_grain, state=state, product=product,
        )
        left_col, right_col = st.columns(2)
        with left_col:
            render_product_bar_chart("DIGITAL", product_series["DIGITAL"])
        with right_col:
            render_product_bar_chart("PHYSICAL", product_series["PHYSICAL"])

        product_comparison = build_channel_comparison_table(
            data, time_grain=time_grain, state=state, product=product,
        )
        with st.expander("📋 DIGITAL vs PHYSICAL comparison"):
            st.dataframe(product_comparison, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------ #
    # Section 3: Applications, Approvals & Funding consistency
    # ------------------------------------------------------------------ #
    st.subheader("Section 3: Applications, Approvals & Funding Consistency")
    application_series = build_application_series(
        data, time_grain=time_grain, state=state,
        product=None if product == PRODUCT_ALL_LABEL else product,
    )
    left_col, right_col = st.columns(2)
    with left_col:
        render_channel_chart_expander("DIGITAL", application_series["DIGITAL"], "outcomes")
    with right_col:
        render_channel_chart_expander("PHYSICAL", application_series["PHYSICAL"], "outcomes")

    # ------------------------------------------------------------------ #
    # Modeling section
    # ------------------------------------------------------------------ #
    st.subheader("Modeling Section")

    # Separate product dropdown for modeling
    model_product_options = get_available_products(data, state)
    model_product = st.selectbox(
        "Product (modeling)", model_product_options, key="model_product",
        help="Select the product to model. Each channel is modeled separately.",
    )

    st.caption(
        "**Model spec — Y-axis (dependent):** Applications by Channel (DIGITAL / PHYSICAL)  ·  "
        "**X-axis (predictors):** all spend tactics + week/bi-weekly time dummies.  "
        "F01 / W_1 / BW_1 is the baseline period (dropped to avoid multicollinearity)."
    )

    modeling_df = prepare_modeling_dataset(
        data, time_grain=time_grain, state=state,
        product=None if model_product == PRODUCT_ALL_LABEL else model_product,
    )
    if modeling_df.empty:
        st.warning("No rows available for the current modeling filter.")
        return

    try:
        model_results = fit_channel_models(modeling_df)
    except Exception as exc:
        st.error(f"Model fitting failed: {exc}")
        return

    left_col, right_col = st.columns(2)
    for column, channel in zip((left_col, right_col), CHANNELS):
        with column:
            if channel not in model_results:
                st.info(f"No {channel.lower()} observations are available for modeling.")
            else:
                render_model_panel(channel, model_results[channel])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("Marketing Analytics and Modeling")
st.caption("Channel-specific analysis and regression for DIGITAL and PHYSICAL applications.")

tabs = st.tabs(["📊 Marketing Spend EDA", "📈 Originations EDA", "🔬 Marketing Analysis"])

with tabs[0]:
    render_tab_marketing_spend()

with tabs[1]:
    render_tab_originations()

with tabs[2]:
    render_tab_marketing_analysis()
