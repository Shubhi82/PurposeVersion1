from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_processing import (
    build_application_series,
    build_channel_comparison_table,
    build_product_spend_series,
    build_tactic_time_series,
    load_modeling_data,
    prepare_modeling_dataset,
)
from modeling import fit_channel_models
from utils import CHANNELS, DEFAULT_DATA_PATH, PRODUCT_ALL_LABEL, TACTIC_COLUMNS, TIME_GRAINS, format_metric, get_available_products


st.set_page_config(page_title="Marketing Analytics and Modeling", layout="wide")


@st.cache_data(show_spinner=False)
def cached_load_data(source: str | Path | bytes) -> pd.DataFrame:
    return load_modeling_data(source)


def load_workbook(uploaded_file, workbook_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return cached_load_data(uploaded_file.getvalue())
    return cached_load_data(workbook_path)


def render_channel_chart(channel: str, chart_data: pd.DataFrame, metric_label: str) -> None:
    st.markdown(f"#### {channel}")
    if chart_data.empty:
        st.info(f"No {channel.lower()} {metric_label.lower()} data is available for the current filter.")
        return
    st.line_chart(chart_data)
    st.dataframe(chart_data.reset_index(), use_container_width=True, hide_index=True)


def render_model_panel(channel: str, result) -> None:
    st.markdown(f"#### {channel} model")
    mape_display = "N/A" if pd.isna(result.mape) else f"{format_metric(result.mape, 2)}%"
    st.metric("MAPE", mape_display)
    st.caption(
        "Dependent variable: "
        f"`Applications_{channel}` | Predictors: channel-specific spend tactics + time dummies "
        f"({', '.join(result.periods_used[:4])}{' ...' if len(result.periods_used) > 4 else ''})."
    )

    chart_frame = result.fitted_frame.set_index("period_label")[["APPLICATIONS", "Predicted_Applications"]]
    st.line_chart(chart_frame)
    st.dataframe(
        result.fitted_frame.rename(
            columns={
                "APPLICATIONS": "Actual_Applications",
                "Predicted_Applications": "Predicted_Applications",
                "TOTAL_SPEND": "Total_Spend",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    tactic_coefficients = result.coefficients.loc[
        result.coefficients["feature"].isin(TACTIC_COLUMNS),
        ["feature", "coefficient", "p_value"],
    ].copy()
    st.markdown("**Coefficient view**")
    st.dataframe(tactic_coefficients, use_container_width=True, hide_index=True)

    if result.narrative:
        st.markdown("**Interpretation**")
        for sentence in result.narrative:
            st.write(sentence)


st.title("Marketing Analytics and Modeling")
st.caption("Channel-specific analysis and regression for DIGITAL and PHYSICAL applications.")

tabs = st.tabs(["Tab 1", "Tab 2", "Marketing Analysis"])

with tabs[0]:
    st.empty()

with tabs[1]:
    st.empty()

with tabs[2]:
    source_col, aggregation_col = st.columns([2, 1])
    with source_col:
        uploaded_file = st.file_uploader("Upload workbook (.xlsx)", type=["xlsx"])
        workbook_path = st.text_input("Workbook path", value=str(DEFAULT_DATA_PATH))
    with aggregation_col:
        time_grain = st.selectbox("Aggregation", TIME_GRAINS)

    try:
        data = load_workbook(uploaded_file, workbook_path)
    except FileNotFoundError:
        st.error("Workbook not found. Upload the file or point the path to a valid `.xlsx` workbook.")
        st.stop()
    except Exception as exc:
        st.error(f"Unable to load workbook: {exc}")
        st.stop()

    state_options = sorted(data["STATE_CD"].dropna().unique().tolist())
    control_col1, control_col2 = st.columns(2)
    with control_col1:
        state = st.selectbox("STATE", state_options)
    with control_col2:
        product = st.selectbox("PRODUCT (optional)", get_available_products(data, state))

    st.subheader("Section 1: Marketing spend consistency")
    tactic_series = build_tactic_time_series(data, time_grain=time_grain, state=state)
    left_col, right_col = st.columns(2)
    with left_col:
        render_channel_chart("DIGITAL", tactic_series["DIGITAL"], "spend")
    with right_col:
        render_channel_chart("PHYSICAL", tactic_series["PHYSICAL"], "spend")

    st.subheader("Section 2: Product-level marketing consistency")
    if product == PRODUCT_ALL_LABEL:
        st.info("Select a specific product to compare DIGITAL and PHYSICAL spend over time.")
    else:
        product_series = build_product_spend_series(
            data,
            time_grain=time_grain,
            state=state,
            product=product,
        )
        product_comparison = build_channel_comparison_table(
            data,
            time_grain=time_grain,
            state=state,
            product=product,
        )
        left_col, right_col = st.columns(2)
        with left_col:
            render_channel_chart("DIGITAL", product_series["DIGITAL"], "product spend")
        with right_col:
            render_channel_chart("PHYSICAL", product_series["PHYSICAL"], "product spend")
        st.markdown("**DIGITAL vs PHYSICAL comparison**")
        st.dataframe(product_comparison, use_container_width=True, hide_index=True)

    st.subheader("Section 3: Applications consistency")
    application_series = build_application_series(
        data,
        time_grain=time_grain,
        state=state,
        product=None if product == PRODUCT_ALL_LABEL else product,
    )
    left_col, right_col = st.columns(2)
    with left_col:
        render_channel_chart("DIGITAL", application_series["DIGITAL"], "applications")
    with right_col:
        render_channel_chart("PHYSICAL", application_series["PHYSICAL"], "applications")

    st.subheader("Modeling section")
    st.caption(
        "Each channel is modeled separately. F01 is the baseline time period and is dropped "
        "from the regression to avoid multicollinearity."
    )

    modeling_df = prepare_modeling_dataset(
        data,
        time_grain=time_grain,
        state=state,
        product=None if product == PRODUCT_ALL_LABEL else product,
    )
    if modeling_df.empty:
        st.warning("No rows are available for the current modeling filter.")
        st.stop()

    try:
        model_results = fit_channel_models(modeling_df)
    except Exception as exc:
        st.error(f"Model fitting failed: {exc}")
        st.stop()

    left_col, right_col = st.columns(2)
    for column, channel in zip((left_col, right_col), CHANNELS):
        with column:
            if channel not in model_results:
                st.info(f"No {channel.lower()} observations are available for modeling.")
            else:
                render_model_panel(channel, model_results[channel])
