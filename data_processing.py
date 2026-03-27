from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from utils import (
    CHANNELS,
    OUTCOME_COLUMNS,
    PRODUCT_ALL_LABEL,
    TACTIC_COLUMNS,
    add_time_columns,
    filter_data,
)


ALL_COMBOS_SHEET = "All_Combos"
NUMERIC_COLUMNS = ["ISO_YEAR", "ISO_WEEK", "CHANNEL_FLAG", *TACTIC_COLUMNS, *OUTCOME_COLUMNS]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_modeling_data(source: str | Path | BinaryIO) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = BytesIO(source)
    df = pd.read_excel(source, sheet_name=ALL_COMBOS_SHEET, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    expected = {"STATE_CD", "PRODUCT_CD", "CHANNEL_CD", "APPLICATIONS", *TACTIC_COLUMNS}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(sorted(missing))}")

    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["STATE_CD"] = df["STATE_CD"].astype(str).str.strip().str.upper()
    df["PRODUCT_CD"] = df["PRODUCT_CD"].astype(str).str.strip()
    df["CHANNEL_CD"] = df["CHANNEL_CD"].astype(str).str.strip().str.upper()
    df = df.loc[df["CHANNEL_CD"].isin(CHANNELS)].copy()

    # Ensure APPROVED and ORIGINATIONS exist (may be absent in older workbooks)
    for col in ["APPROVED", "ORIGINATIONS"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def load_marketing_spend_raw(source) -> pd.DataFrame:
    """Load the raw Marketing_Spend_Data CSV for EDA."""
    if isinstance(source, bytes):
        source = BytesIO(source)
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    df.drop_duplicates(inplace=True)

    if "BUSINESS_DATE" in df.columns:
        df["BUSINESS_DATE"] = pd.to_datetime(df["BUSINESS_DATE"], errors="coerce")
        df["ISO_YEAR"] = df["BUSINESS_DATE"].dt.isocalendar().year.astype("Int64")
        df["ISO_WEEK"] = df["BUSINESS_DATE"].dt.isocalendar().week.astype("Int64")
        df["MONTH"] = df["BUSINESS_DATE"].dt.month

    if "TOTAL_COST" in df.columns:
        df["TOTAL_COST"] = pd.to_numeric(df["TOTAL_COST"], errors="coerce").fillna(0.0)

    return df


def build_marketing_spend_fallback(model_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["ISO_YEAR", "ISO_WEEK", "STATE_CD", "PRODUCT_CD", "CHANNEL_CD"]
    available = [column for column in base_columns + TACTIC_COLUMNS if column in model_df.columns]
    fallback = model_df[available].copy()
    fallback = fallback.melt(
        id_vars=[column for column in base_columns if column in fallback.columns],
        value_vars=[column for column in TACTIC_COLUMNS if column in fallback.columns],
        var_name="DETAIL_TACTIC",
        value_name="TOTAL_COST",
    )
    fallback["TOTAL_COST"] = pd.to_numeric(fallback["TOTAL_COST"], errors="coerce").fillna(0.0)
    fallback = fallback.loc[fallback["TOTAL_COST"] > 0].reset_index(drop=True)
    fallback.attrs["source_label"] = "fallback_modeling_workbook"
    return fallback


def load_marketing_spend_data(source, fallback_source: str | Path | BinaryIO | None = None) -> pd.DataFrame:
    try:
        return load_marketing_spend_raw(source)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError, ValueError):
        if fallback_source is None:
            raise
        model_df = load_modeling_data(fallback_source)
        return build_marketing_spend_fallback(model_df)


def load_originations_raw(source) -> pd.DataFrame:
    """
    Load the raw Originations data for EDA.
    Supports both CSV and Excel (.xlsx) files.
    """

    # Handle uploaded files (bytes)
    if isinstance(source, bytes):
        source = BytesIO(source)

    # Detect file type and load accordingly
    if isinstance(source, (str, Path)):
        file_path = str(source).lower()

        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(source, engine="openpyxl")
        elif file_path.endswith(".csv"):
            df = pd.read_csv(source)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or Excel file.")

    else:
        # If it's already a buffer (e.g., BytesIO), try Excel first then CSV
        try:
            df = pd.read_excel(source, engine="openpyxl")
        except Exception:
            df = pd.read_csv(source)

    # ---------------------------------------------------
    # Standard cleaning
    # ---------------------------------------------------
    df.columns = [str(c).strip() for c in df.columns]

    # Date processing
    if "APPLICATION_DT" in df.columns:
        df["APPLICATION_DT"] = pd.to_datetime(df["APPLICATION_DT"], errors="coerce")
        df["ISO_YEAR"] = df["APPLICATION_DT"].dt.isocalendar().year.astype("Int64")
        df["ISO_WEEK"] = df["APPLICATION_DT"].dt.isocalendar().week.astype("Int64")
        df["MONTH"] = df["APPLICATION_DT"].dt.month

    # Rename column if needed
    if "PRODUCT_CODE" in df.columns and "PRODUCT_CD" not in df.columns:
        df = df.rename(columns={"PRODUCT_CODE": "PRODUCT_CD"})

    # Numeric conversions
    for col in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def build_originations_fallback(model_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["ISO_YEAR", "ISO_WEEK", "STATE_CD", "PRODUCT_CD", "CHANNEL_CD"]
    available = [column for column in base_columns + OUTCOME_COLUMNS if column in model_df.columns]
    fallback = model_df[available].copy()
    for column in OUTCOME_COLUMNS:
        if column not in fallback.columns:
            fallback[column] = 0.0
    fallback.attrs["source_label"] = "fallback_modeling_workbook"
    return fallback


def load_originations_data(source, fallback_source: str | Path | BinaryIO | None = None) -> pd.DataFrame:
    try:
        return load_originations_raw(source)
    except (BadZipFile, pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError, ValueError):
        if fallback_source is None:
            raise
        model_df = load_modeling_data(fallback_source)
        return build_originations_fallback(model_df)


# ---------------------------------------------------------------------------
# EDA aggregations — Marketing Spend
# ---------------------------------------------------------------------------

def ms_spend_by_tactic(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("DETAIL_TACTIC")["TOTAL_COST"]
        .sum()
        .reset_index()
        .rename(columns={"DETAIL_TACTIC": "Tactic", "TOTAL_COST": "Total Spend ($)"})
        .sort_values("Total Spend ($)", ascending=False)
    )


def ms_spend_by_state(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("STATE_CD")["TOTAL_COST"]
        .sum()
        .reset_index()
        .rename(columns={"STATE_CD": "State", "TOTAL_COST": "Total Spend ($)"})
        .sort_values("Total Spend ($)", ascending=False)
    )


def ms_spend_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("CHANNEL_CD")["TOTAL_COST"]
        .sum()
        .reset_index()
        .rename(columns={"CHANNEL_CD": "Channel", "TOTAL_COST": "Total Spend ($)"})
        .sort_values("Total Spend ($)", ascending=False)
    )


def ms_spend_by_product(df: pd.DataFrame) -> pd.DataFrame:
    col = "PRODUCT_CD" if "PRODUCT_CD" in df.columns else "H_TACTIC"
    return (
        df.groupby(col)["TOTAL_COST"]
        .sum()
        .reset_index()
        .rename(columns={col: "Product", "TOTAL_COST": "Total Spend ($)"})
        .sort_values("Total Spend ($)", ascending=False)
    )


def ms_spend_over_time(df: pd.DataFrame) -> pd.DataFrame:
    if "ISO_YEAR" not in df.columns or "ISO_WEEK" not in df.columns:
        return pd.DataFrame()
    grp = (
        df.groupby(["ISO_YEAR", "ISO_WEEK"])["TOTAL_COST"]
        .sum()
        .reset_index()
    )
    grp["period"] = grp["ISO_YEAR"].astype(str) + "-W" + grp["ISO_WEEK"].astype(str).str.zfill(2)
    return grp.sort_values(["ISO_YEAR", "ISO_WEEK"])


def ms_tactic_channel_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["CHANNEL_CD", "DETAIL_TACTIC"])["TOTAL_COST"]
        .sum()
        .reset_index()
        .pivot(index="DETAIL_TACTIC", columns="CHANNEL_CD", values="TOTAL_COST")
        .fillna(0)
        .reset_index()
        .rename(columns={"DETAIL_TACTIC": "Tactic"})
    )


def ms_spend_by_state_tactic(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["STATE_CD", "DETAIL_TACTIC"])["TOTAL_COST"]
        .sum()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# EDA aggregations — Originations
# ---------------------------------------------------------------------------

def od_metrics_by_state(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in df.columns]
    return (
        df.groupby("STATE_CD")[cols]
        .sum()
        .reset_index()
        .rename(columns={"STATE_CD": "State"})
        .sort_values("APPLICATIONS", ascending=False)
    )


def od_metrics_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in df.columns]
    return (
        df.groupby("CHANNEL_CD")[cols]
        .sum()
        .reset_index()
        .rename(columns={"CHANNEL_CD": "Channel"})
    )


def od_metrics_by_product(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in df.columns]
    col = "PRODUCT_CD" if "PRODUCT_CD" in df.columns else "H_TACTIC"
    return (
        df.groupby(col)[cols]
        .sum()
        .reset_index()
        .rename(columns={col: "Product"})
        .sort_values("APPLICATIONS", ascending=False)
    )


def od_metrics_over_time(df: pd.DataFrame) -> pd.DataFrame:
    if "ISO_YEAR" not in df.columns or "ISO_WEEK" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in df.columns]
    grp = (
        df.groupby(["ISO_YEAR", "ISO_WEEK"])[cols]
        .sum()
        .reset_index()
    )
    grp["period"] = grp["ISO_YEAR"].astype(str) + "-W" + grp["ISO_WEEK"].astype(str).str.zfill(2)
    return grp.sort_values(["ISO_YEAR", "ISO_WEEK"])


def od_funnel_by_state(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in df.columns]
    grp = df.groupby("STATE_CD")[cols].sum().reset_index()
    if "APPROVED" in grp.columns and "APPLICATIONS" in grp.columns:
        grp["Approval Rate (%)"] = (grp["APPROVED"] / grp["APPLICATIONS"].replace(0, np.nan) * 100).round(1)
    if "ORIGINATIONS" in grp.columns and "APPROVED" in grp.columns:
        grp["Funding Rate (%)"] = (grp["ORIGINATIONS"] / grp["APPROVED"].replace(0, np.nan) * 100).round(1)
    return grp.rename(columns={"STATE_CD": "State"})


def od_channel_state_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["APPLICATIONS", "APPROVED", "ORIGINATIONS"] if c in df.columns]
    return (
        df.groupby(["STATE_CD", "CHANNEL_CD"])[cols]
        .sum()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Modeling dataset builders (from the loaded Excel workbook)
# ---------------------------------------------------------------------------

def split_by_channel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {ch: df.loc[df["CHANNEL_CD"] == ch].copy() for ch in CHANNELS}


def aggregate_for_analysis(
    df: pd.DataFrame,
    time_grain: str,
    state: str,
    product: str | None = None,
) -> pd.DataFrame:
    scoped = filter_data(df, state=state, product=product)
    scoped = add_time_columns(scoped, time_grain)
    agg_cols = TACTIC_COLUMNS + [c for c in OUTCOME_COLUMNS if c in scoped.columns]
    group_cols = ["period_id", "period_label", "period_start", "period_number", "CHANNEL_CD"]
    aggregated = (
        scoped.groupby(group_cols, dropna=False)[agg_cols]
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
        chart_data[channel] = channel_data.set_index("period_label")[TACTIC_COLUMNS]
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
        channel_data = aggregated.loc[aggregated["CHANNEL_CD"] == channel].copy()
        chart_data[channel] = channel_data.set_index("period_label")[TACTIC_COLUMNS + ["TOTAL_SPEND"]]
    return chart_data


def build_application_series(
    df: pd.DataFrame,
    time_grain: str,
    state: str,
    product: str | None = None,
) -> dict[str, pd.DataFrame]:
    aggregated = aggregate_for_analysis(df, time_grain=time_grain, state=state, product=product)
    outcome_cols = [c for c in OUTCOME_COLUMNS if c in aggregated.columns]
    chart_data: dict[str, pd.DataFrame] = {}
    for channel in CHANNELS:
        channel_data = aggregated.loc[aggregated["CHANNEL_CD"] == channel].copy()
        chart_data[channel] = channel_data.set_index("period_label")[outcome_cols]
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
    value_cols = ["TOTAL_SPEND"] + [c for c in OUTCOME_COLUMNS if c in aggregated.columns]
    comparison = (
        aggregated.pivot_table(
            index="period_label",
            columns="CHANNEL_CD",
            values=value_cols,
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
    modeling_frame["time_bucket"] = (
        "F" + modeling_frame["period_number"].astype(int).astype(str).str.zfill(2)
    )
    time_dummies = pd.get_dummies(modeling_frame["time_bucket"], dtype=float)
    if "F01" in time_dummies.columns:
        time_dummies = time_dummies.drop(columns="F01")

    return pd.concat(
        [modeling_frame.reset_index(drop=True), time_dummies.reset_index(drop=True)], axis=1
    )
