from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from utils import CHANNELS, TACTIC_COLUMNS, add_time_columns, filter_data


ALL_COMBOS_SHEET = "All_Combos"
NUMERIC_COLUMNS = ["ISO_YEAR", "ISO_WEEK", "ISO_MONTH", "Channel flag", *TACTIC_COLUMNS, "APPLICATIONS"]


def load_modeling_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = BytesIO(source)
    df = pd.read_excel(source, sheet_name=ALL_COMBOS_SHEET, engine="openpyxl")
    df.columns = [str(column).strip() for column in df.columns]

    expected_columns = {"STATE_CD", "PRODUCT_CD", "CHANNEL_CD", "APPLICATIONS", *TACTIC_COLUMNS}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing expected columns in workbook: {missing_list}")

    available_numeric = [column for column in NUMERIC_COLUMNS if column in df.columns]
    for column in available_numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["STATE_CD"] = df["STATE_CD"].astype(str).str.strip().str.upper()
    df["PRODUCT_CD"] = df["PRODUCT_CD"].astype(str).str.strip()
    df["CHANNEL_CD"] = df["CHANNEL_CD"].astype(str).str.strip().str.upper()
    df = df.loc[df["CHANNEL_CD"].isin(CHANNELS)].copy()

    return df


def split_by_channel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {channel: df.loc[df["CHANNEL_CD"] == channel].copy() for channel in CHANNELS}


def aggregate_for_analysis(
    df: pd.DataFrame,
    time_grain: str,
    state: str,
    product: str | None = None,
) -> pd.DataFrame:
    scoped = filter_data(df, state=state, product=product)
    scoped = add_time_columns(scoped, time_grain)
    group_columns = ["period_id", "period_label", "period_start", "period_number", "CHANNEL_CD"]
    aggregated = (
        scoped.groupby(group_columns, dropna=False)[TACTIC_COLUMNS + ["APPLICATIONS"]]
        .sum()
        .reset_index()
        .sort_values(["period_start", "CHANNEL_CD"])
    )
    aggregated["TOTAL_SPEND"] = aggregated[TACTIC_COLUMNS].sum(axis=1)
    return aggregated


def build_tactic_time_series(
    df: pd.DataFrame,
    time_grain: str,
    state: str,
    product: str | None = None,
) -> dict[str, pd.DataFrame]:
    aggregated = aggregate_for_analysis(df, time_grain=time_grain, state=state, product=product)
    chart_data: dict[str, pd.DataFrame] = {}
    for channel in CHANNELS:
        channel_data = aggregated.loc[aggregated["CHANNEL_CD"] == channel].copy()
        if channel_data.empty:
            chart_data[channel] = pd.DataFrame()
            continue
        pivoted = channel_data.set_index("period_label")[TACTIC_COLUMNS]
        chart_data[channel] = pivoted
    return chart_data


def build_product_spend_series(
    df: pd.DataFrame,
    time_grain: str,
    state: str,
    product: str,
) -> dict[str, pd.DataFrame]:
    aggregated = aggregate_for_analysis(df, time_grain=time_grain, state=state, product=product)
    chart_data: dict[str, pd.DataFrame] = {}
    for channel in CHANNELS:
        channel_data = aggregated.loc[aggregated["CHANNEL_CD"] == channel, ["period_label", "TOTAL_SPEND"]].copy()
        chart_data[channel] = channel_data.set_index("period_label")
    return chart_data


def build_application_series(
    df: pd.DataFrame,
    time_grain: str,
    state: str,
    product: str | None = None,
) -> dict[str, pd.DataFrame]:
    aggregated = aggregate_for_analysis(df, time_grain=time_grain, state=state, product=product)
    chart_data: dict[str, pd.DataFrame] = {}
    for channel in CHANNELS:
        channel_data = aggregated.loc[aggregated["CHANNEL_CD"] == channel, ["period_label", "APPLICATIONS"]].copy()
        chart_data[channel] = channel_data.set_index("period_label")
    return chart_data


def build_channel_comparison_table(
    df: pd.DataFrame,
    time_grain: str,
    state: str,
    product: str | None = None,
) -> pd.DataFrame:
    aggregated = aggregate_for_analysis(df, time_grain=time_grain, state=state, product=product)
    if aggregated.empty:
        return pd.DataFrame()

    comparison = (
        aggregated.pivot_table(
            index="period_label",
            columns="CHANNEL_CD",
            values=["TOTAL_SPEND", "APPLICATIONS"],
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )
    comparison.columns = [f"{metric}_{channel}" for metric, channel in comparison.columns]
    return comparison.reset_index()


def prepare_modeling_dataset(
    df: pd.DataFrame,
    time_grain: str,
    state: str,
    product: str | None = None,
) -> pd.DataFrame:
    aggregated = aggregate_for_analysis(df, time_grain=time_grain, state=state, product=product)
    if aggregated.empty:
        return aggregated

    modeling_frame = aggregated.copy()
    modeling_frame["time_bucket"] = "F" + modeling_frame["period_number"].astype(int).astype(str).str.zfill(2)

    time_dummies = pd.get_dummies(modeling_frame["time_bucket"], dtype=float)
    if "F01" in time_dummies.columns:
        time_dummies = time_dummies.drop(columns="F01")

    return pd.concat([modeling_frame.reset_index(drop=True), time_dummies.reset_index(drop=True)], axis=1)
