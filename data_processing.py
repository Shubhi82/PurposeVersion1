from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from utils import (
    CHANNELS,
    DM_DATA_PATH,
    MARKETING_SPEND_PATH,
    ORIGINATIONS_V5_PATH,
    OUTCOME_COLUMNS,
    PRODUCT_ALL_LABEL,
    TACTIC_COLUMNS,
    add_time_columns,
    expand_rollup_product,
    filter_data,
    is_rolled_up_product,
)


ALL_COMBOS_SHEET = "All_Combos"
MODELING_SHEET_MAP = {
    "Weekly": "Weekly_Data",
    "Fortnight": "Fortnightly_Data",
    "Fortnightly": "Fortnightly_Data",
}
NUMERIC_COLUMNS = ["ISO_YEAR", "ISO_WEEK", "CHANNEL_FLAG", *TACTIC_COLUMNS, *OUTCOME_COLUMNS]
WEEK_DUMMY_COLUMNS = [f"W_{week}" for week in range(1, 53)]
BIWEEK_DUMMY_COLUMNS = [f"BW_{bw}" for bw in range(1, 27)]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_modeling_data(source: str | Path | BinaryIO, time_grain: str = "Weekly") -> pd.DataFrame:
    if isinstance(source, bytes):
        source = BytesIO(source)
    workbook = pd.ExcelFile(source, engine="openpyxl")
    preferred_sheet = MODELING_SHEET_MAP.get(time_grain, ALL_COMBOS_SHEET)
    if preferred_sheet in workbook.sheet_names:
        sheet_name = preferred_sheet
    elif ALL_COMBOS_SHEET in workbook.sheet_names:
        sheet_name = ALL_COMBOS_SHEET
    else:
        sheet_name = workbook.sheet_names[0]

    df = pd.read_excel(workbook, sheet_name=sheet_name, engine="openpyxl")
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

    if "TOTAL_SPEND" not in df.columns:
        available_tactics = [col for col in TACTIC_COLUMNS if col in df.columns]
        df["TOTAL_SPEND"] = df[available_tactics].sum(axis=1) if available_tactics else 0.0
    if "time_grain" not in df.columns:
        df["time_grain"] = str(time_grain).lower()

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
        model_df = load_modeling_data(fallback_source, time_grain="Weekly")
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
        model_df = load_modeling_data(fallback_source, time_grain="Weekly")
        return build_originations_fallback(model_df)


def load_dm_data(source: str | Path | BinaryIO = DM_DATA_PATH) -> pd.DataFrame:
    """Load Direct Mail data as a separate input from the core spend/origination files."""
    if isinstance(source, bytes):
        source = BytesIO(source)
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "ADDRESS_STATE": "STATE_CD",
        "OFFER_PRODUCT": "PRODUCT_CD",
        "OFFER_CHANNEL": "DM_CHANNEL_CD",
        "EXPECTED_INHOME_DATE": "INHOME_DATE",
    }
    df = df.rename(columns={src: dest for src, dest in rename_map.items() if src in df.columns})

    if "INHOME_DATE" in df.columns:
        df["INHOME_DATE"] = pd.to_datetime(df["INHOME_DATE"], errors="coerce")
        df["ISO_YEAR"] = df["INHOME_DATE"].dt.isocalendar().year.astype("Int64")
        df["ISO_WEEK"] = df["INHOME_DATE"].dt.isocalendar().week.astype("Int64")

    if "STATE_CD" in df.columns:
        df["STATE_CD"] = df["STATE_CD"].astype(str).str.strip().str.upper()
    if "PRODUCT_CD" in df.columns:
        df["PRODUCT_CD"] = df["PRODUCT_CD"].astype(str).str.strip()
    if "DM_CHANNEL_CD" in df.columns:
        df["DM_CHANNEL_CD"] = df["DM_CHANNEL_CD"].astype(str).str.strip().str.upper()

    numeric_cols = [
        "UNIQUE_RESERVATION_CODE_COUNT",
        "UNIQUE_APPLICANTS",
        "UNIQUE_APPROVED",
        "UNIQUE_FUNDED",
        "RESPONSE_RATE_PCT",
        "APPROVAL_RATE_PCT",
        "FUND_RATE_PCT",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def filter_rolled_up_products(df: pd.DataFrame, product_col: str = "PRODUCT_CD") -> pd.DataFrame:
    if product_col not in df.columns:
        return df.copy()
    out = df.copy()
    return out.loc[out[product_col].astype(str).str.strip().apply(is_rolled_up_product)].copy()


def exclude_direct_mail_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Direct Mail rows from marketing spend data when present."""
    out = df.copy()
    masks = []
    for col in ["DETAIL_TACTIC", "H_TACTIC"]:
        if col in out.columns:
            masks.append(out[col].astype(str).str.contains("direct mail", case=False, na=False))
    if not masks:
        return out
    dm_mask = masks[0]
    for mask in masks[1:]:
        dm_mask = dm_mask | mask
    return out.loc[~dm_mask].copy()


def summarize_dm_data(
    df: pd.DataFrame,
    state: str | None = None,
    product: str | None = None,
    time_grain: str = "Weekly",
) -> pd.DataFrame:
    dm = df.copy()
    if state:
        dm = dm.loc[dm["STATE_CD"] == state].copy()
    if product and product != PRODUCT_ALL_LABEL:
        mapped_products = expand_rollup_product(product)
        if mapped_products:
            dm = dm.loc[dm["PRODUCT_CD"].isin(mapped_products)].copy()

    dm = dm.loc[dm["ISO_YEAR"].notna() & dm["ISO_WEEK"].notna()].copy()
    dm["ISO_YEAR"] = dm["ISO_YEAR"].astype(int)
    dm["ISO_WEEK"] = dm["ISO_WEEK"].astype(int)
    dm = add_time_columns(dm, "Fortnight" if time_grain in {"Fortnight", "Fortnightly"} else "Weekly")

    metrics = [
        "UNIQUE_RESERVATION_CODE_COUNT",
        "UNIQUE_APPLICANTS",
        "UNIQUE_APPROVED",
        "UNIQUE_FUNDED",
    ]
    available = [col for col in metrics if col in dm.columns]
    if not available:
        return pd.DataFrame()

    summary = (
        dm.groupby(["period_label", "DM_CHANNEL_CD"], dropna=False)[available]
        .sum()
        .reset_index()
        .sort_values(["period_label", "DM_CHANNEL_CD"])
    )
    return summary


def prepare_raw_mmm_dataset(
    marketing_df: pd.DataFrame,
    originations_df: pd.DataFrame,
    state: str | None = None,
    product: str | None = None,
    time_grain: str = "Weekly",
) -> pd.DataFrame:
    """Build a weekly or fortnight MMM dataset from raw marketing and originations files."""
    ms = marketing_df.copy()
    od = originations_df.copy()

    for frame in (ms, od):
        frame["STATE_CD"] = frame["STATE_CD"].astype(str).str.strip().str.upper()
        frame["CHANNEL_CD"] = frame["CHANNEL_CD"].astype(str).str.strip().str.upper()
        if "PRODUCT_CD" in frame.columns:
            frame["PRODUCT_CD"] = frame["PRODUCT_CD"].astype(str).str.strip()

    ms = ms.loc[ms["CHANNEL_CD"].isin(CHANNELS)].copy()
    od = od.loc[od["CHANNEL_CD"].isin(CHANNELS)].copy()

    if state:
        ms = ms.loc[ms["STATE_CD"] == state]
        od = od.loc[od["STATE_CD"] == state]
    if product and product != PRODUCT_ALL_LABEL:
        ms = ms.loc[ms["PRODUCT_CD"] == product]
        od = od.loc[od["PRODUCT_CD"] == product]

    ms = ms.loc[ms["DETAIL_TACTIC"].isin(TACTIC_COLUMNS)].copy()
    spend_keys = ["ISO_YEAR", "ISO_WEEK", "STATE_CD", "CHANNEL_CD", "PRODUCT_CD"]
    spend = (
        ms.pivot_table(
            index=spend_keys,
            columns="DETAIL_TACTIC",
            values="TOTAL_COST",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    spend.columns.name = None
    for tactic in TACTIC_COLUMNS:
        if tactic not in spend.columns:
            spend[tactic] = 0.0

    outcome_cols = [column for column in OUTCOME_COLUMNS if column in od.columns]
    outcomes = (
        od.groupby(spend_keys, dropna=False)[outcome_cols]
        .sum()
        .reset_index()
    )
    for column in OUTCOME_COLUMNS:
        if column not in outcomes.columns:
            outcomes[column] = 0.0

    merged = outcomes.merge(spend, on=spend_keys, how="outer").fillna(0.0)
    for column in TACTIC_COLUMNS + OUTCOME_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0

    merged["ISO_YEAR"] = pd.to_numeric(merged["ISO_YEAR"], errors="coerce").fillna(0).astype(int)
    merged["ISO_WEEK"] = pd.to_numeric(merged["ISO_WEEK"], errors="coerce").fillna(0).astype(int)
    merged = merged.loc[(merged["ISO_YEAR"] > 0) & (merged["ISO_WEEK"] > 0)].copy()

    if time_grain in {"Fortnight", "Fortnightly"}:
        merged["FORTNIGHT"] = np.minimum(((merged["ISO_WEEK"] - 1) // 2) + 1, 26).astype(int)
        agg_columns = [*TACTIC_COLUMNS, *OUTCOME_COLUMNS]
        merged = (
            merged.groupby(
                ["ISO_YEAR", "FORTNIGHT", "STATE_CD", "CHANNEL_CD", "PRODUCT_CD"],
                dropna=False,
            )[agg_columns]
            .sum()
            .reset_index()
        )
        fortnight_dummies = pd.get_dummies(merged["FORTNIGHT"], prefix="BW")
        fortnight_dummies = fortnight_dummies.reindex(columns=BIWEEK_DUMMY_COLUMNS, fill_value=0).astype(int)
        merged = pd.concat([merged.reset_index(drop=True), fortnight_dummies.reset_index(drop=True)], axis=1)
        merged["TOTAL_SPEND"] = merged[TACTIC_COLUMNS].sum(axis=1)
        merged["period_label"] = (
            merged["ISO_YEAR"].astype(str) + "-BW" + merged["FORTNIGHT"].astype(str).str.zfill(2)
        )
        merged = merged.sort_values(
            ["STATE_CD", "CHANNEL_CD", "PRODUCT_CD", "ISO_YEAR", "FORTNIGHT"]
        ).reset_index(drop=True)
        return merged

    week_dummies = pd.get_dummies(merged["ISO_WEEK"], prefix="W")
    week_dummies = week_dummies.reindex(columns=WEEK_DUMMY_COLUMNS, fill_value=0).astype(int)
    merged = pd.concat([merged.reset_index(drop=True), week_dummies.reset_index(drop=True)], axis=1)

    merged["TOTAL_SPEND"] = merged[TACTIC_COLUMNS].sum(axis=1)
    merged["period_label"] = (
        merged["ISO_YEAR"].astype(str) + "-W" + merged["ISO_WEEK"].astype(str).str.zfill(2)
    )
    merged = merged.sort_values(["STATE_CD", "CHANNEL_CD", "PRODUCT_CD", "ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)

    return merged


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


# ---------------------------------------------------------------------------
# Version 5 — Live Regression Pipeline
# ---------------------------------------------------------------------------

TACTIC_COLS_V5 = ["DSP", "LeadGen", "Paid Search", "Paid Social", "Prescreen", "Referrals"]

STATE_TO_DIVISION = {
    "CT": "New England", "ME": "New England", "MA": "New England", "NH": "New England",
    "RI": "New England", "VT": "New England",
    "NJ": "Middle Atlantic", "NY": "Middle Atlantic", "PA": "Middle Atlantic",
    "IL": "East North Central", "IN": "East North Central", "MI": "East North Central",
    "OH": "East North Central", "WI": "East North Central",
    "IA": "West North Central", "KS": "West North Central", "MN": "West North Central",
    "MO": "West North Central", "NE": "West North Central", "ND": "West North Central",
    "SD": "West North Central",
    "DE": "South Atlantic", "FL": "South Atlantic", "GA": "South Atlantic",
    "MD": "South Atlantic", "NC": "South Atlantic", "SC": "South Atlantic",
    "VA": "South Atlantic", "WV": "South Atlantic", "DC": "South Atlantic",
    "AL": "East South Central", "KY": "East South Central", "MS": "East South Central",
    "TN": "East South Central",
    "AR": "West South Central", "LA": "West South Central", "OK": "West South Central",
    "TX": "West South Central",
    "AZ": "Mountain", "CO": "Mountain", "ID": "Mountain", "MT": "Mountain",
    "NV": "Mountain", "NM": "Mountain", "UT": "Mountain", "WY": "Mountain",
    "AK": "Pacific", "CA": "Pacific", "HI": "Pacific", "OR": "Pacific", "WA": "Pacific",
}

DUMMY_FAMILIES_V5 = {
    "f_dummy": {
        "features": lambda: [f"F_{i}" for i in range(1, 26)],
        "scaler": "minmax",
    },
    "weekly": {
        "features": lambda: [f"W_{i}" for i in range(2, 53)],
        "scaler": "minmax",
    },
    "fourier": {
        "features": lambda: ["sin_1", "cos_1", "sin_2", "cos_2"],
        "scaler": "standard",
    },
}


def build_modeling_frame(channel: str, product: str | None = None) -> pd.DataFrame:
    """
    Build weekly modeling frame for DIGITAL or PHYSICAL channel.
    Optionally filter by rolled-up product (pass product name or None for all).
    Replicates notebook 00 exactly.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: F401 (ensure available)

    ms = pd.read_csv(str(MARKETING_SPEND_PATH)).drop_duplicates()
    dm = pd.read_csv(str(DM_DATA_PATH)).drop_duplicates()
    orig = pd.read_excel(str(ORIGINATIONS_V5_PATH))

    # Date parsing and ISO week assignment
    ms["BUSINESS_DATE"] = pd.to_datetime(ms["BUSINESS_DATE"])
    orig["APPLICATION_DT"] = pd.to_datetime(orig["APPLICATION_DT"])
    dm_date = pd.to_datetime(dm["EXPECTED_INHOME_DATE"])
    for df_, col in [(ms, "BUSINESS_DATE"), (orig, "APPLICATION_DT")]:
        df_["ISO_YEAR"] = df_[col].dt.isocalendar().year
        df_["ISO_WEEK"] = df_[col].dt.isocalendar().week
    dm["ISO_YEAR"] = dm_date.dt.isocalendar().year
    dm["ISO_WEEK"] = dm_date.dt.isocalendar().week

    # Marketing spend: pivot to wide (one column per tactic)
    ms_w = (
        ms.groupby(["ISO_YEAR", "ISO_WEEK", "DETAIL_TACTIC", "STATE_CD"])["TOTAL_COST"]
        .sum()
        .reset_index()
    )
    ms_wide = ms_w.pivot_table(
        index=["ISO_YEAR", "ISO_WEEK", "STATE_CD"],
        columns="DETAIL_TACTIC",
        values="TOTAL_COST",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    ms_wide.columns.name = None

    # Originations: optionally filter by rolled-up product, then aggregate by channel
    if product and product != PRODUCT_ALL_LABEL and "PRODUCT_CODE" in orig.columns:
        product_codes = expand_rollup_product(product)
        if product_codes:
            orig = orig[orig["PRODUCT_CODE"].isin(product_codes)].copy()

    orig_clean = orig.drop(
        columns=[c for c in ["H_TACTIC", "ORIGINATION_DT", "DETAIL_TACTIC", "PRODUCT_CODE"] if c in orig.columns]
    )
    od_w = (
        orig_clean.groupby(["ISO_YEAR", "ISO_WEEK", "STATE_CD", "CHANNEL_CD"])
        .agg({"APPLICATIONS": "sum"})
        .reset_index()
    )

    # Direct mail: split DIGITAL (ONLINE) vs PHYSICAL (OMNI + STORE)
    dm_clean = dm[["ADDRESS_STATE", "OFFER_CHANNEL", "UNIQUE_APPLICANTS"]].copy()
    dm_clean["ISO_YEAR"] = dm_date.dt.isocalendar().year
    dm_clean["ISO_WEEK"] = dm_date.dt.isocalendar().week
    dm_clean = dm_clean.rename(columns={
        "ADDRESS_STATE": "STATE_CD",
        "UNIQUE_APPLICANTS": "DM_APPLICATIONS",
        "OFFER_CHANNEL": "DM_CHANNEL",
    })
    dm_w = (
        dm_clean.groupby(["ISO_YEAR", "ISO_WEEK", "STATE_CD", "DM_CHANNEL"])
        .agg({"DM_APPLICATIONS": "sum"})
        .reset_index()
    )

    if channel == "DIGITAL":
        dm_ch = dm_w[dm_w["DM_CHANNEL"] == "ONLINE"]
    else:
        dm_ch = dm_w[dm_w["DM_CHANNEL"] != "ONLINE"]
    dm_ch = (
        dm_ch.groupby(["ISO_YEAR", "ISO_WEEK", "STATE_CD"])
        .agg({"DM_APPLICATIONS": "sum"})
        .reset_index()
    )

    # Filter originations to channel
    od_ch = od_w[od_w["CHANNEL_CD"] == channel].copy()

    # Merge originations + DM, compute NON_DM_APPLICATIONS
    merged = pd.merge(od_ch, dm_ch, on=["ISO_YEAR", "ISO_WEEK", "STATE_CD"], how="left")
    merged["NON_DM_APPLICATIONS"] = np.maximum(
        0, merged["APPLICATIONS"] - merged["DM_APPLICATIONS"].fillna(0)
    )

    # Final merge with spend
    df = pd.merge(
        ms_wide,
        merged[["ISO_YEAR", "ISO_WEEK", "STATE_CD", "APPLICATIONS", "NON_DM_APPLICATIONS"]],
        on=["ISO_YEAR", "ISO_WEEK", "STATE_CD"],
        how="inner",
    ).fillna(0)

    # Seasonality: fortnightly dummies F_0..F_25
    f_dummies = pd.get_dummies((df["ISO_WEEK"] - 1) // 2, prefix="F", dtype=int)
    df = pd.concat([df, f_dummies], axis=1)
    for i in range(26):
        if f"F_{i}" not in df.columns:
            df[f"F_{i}"] = 0

    # Seasonality: weekly dummies W_1..W_52
    w_dummies = pd.get_dummies(df["ISO_WEEK"], prefix="W", dtype=int)
    df = pd.concat([df, w_dummies], axis=1)
    for i in range(1, 53):
        if f"W_{i}" not in df.columns:
            df[f"W_{i}"] = 0

    # Seasonality: Fourier terms
    P = 52.0
    for k in [1, 2]:
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * df["ISO_WEEK"] / P)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * df["ISO_WEEK"] / P)

    df["Division"] = df["STATE_CD"].map(STATE_TO_DIVISION)
    return df


def fit_model_config(
    entity_df: pd.DataFrame,
    channel: str,
    model_type: str,
    dummy_family: str,
    train_years: list = None,
    n_test_weeks: int = 8,
) -> dict | None:
    """
    Fit one model configuration exactly matching the offline notebook pipeline.
    Returns a diagnostics dict or None if insufficient data.
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from scipy.optimize import nnls

    if train_years is None:
        train_years = [2024, 2025]

    tactics = TACTIC_COLS_V5
    family = DUMMY_FAMILIES_V5[dummy_family]
    seasonal = family["features"]()
    scaler_type = family["scaler"]
    features = [f for f in tactics + seasonal if f in entity_df.columns]
    if not features:
        return None

    train = entity_df[entity_df["ISO_YEAR"].isin(train_years)].copy()
    next_years = sorted([y for y in entity_df["ISO_YEAR"].unique() if y not in train_years])
    if not next_years:
        return None
    next_year = next_years[0]
    test_all = entity_df[entity_df["ISO_YEAR"] == next_year].copy().sort_values(["ISO_YEAR", "ISO_WEEK"])
    test = test_all[test_all["ISO_WEEK"] <= n_test_weeks]

    if len(train) < 10 or len(test) == 0:
        return None

    X_tr = train[features].values.astype(float)
    y_tr = train["NON_DM_APPLICATIONS"].values.astype(float)
    X_te = test[features].values.astype(float)
    y_te = test["NON_DM_APPLICATIONS"].values.astype(float)

    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    if model_type == "NNLS":
        coefs, _ = nnls(X_tr_s, y_tr)
        yhat_tr = X_tr_s @ coefs
        yhat_te = X_te_s @ coefs
    else:
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X_tr_s, y_tr)
        yhat_tr = lr.predict(X_tr_s)
        yhat_te = lr.predict(X_te_s)
        coefs = lr.coef_

    n, p = len(y_tr), len(features)
    r2 = float(r2_score(y_tr, yhat_tr))
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    mae = float(mean_absolute_error(y_tr, yhat_tr))
    mask = y_tr != 0
    mape = float(np.mean(np.abs((y_tr[mask] - yhat_tr[mask]) / y_tr[mask])) * 100) if mask.any() else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_tr, yhat_tr)))
    test_r2 = float(r2_score(y_te, yhat_te)) if len(y_te) > 0 else np.nan
    sse = float(np.sum((y_tr - yhat_tr) ** 2))
    aic = n * np.log(sse / n) + 2 * p
    bic = n * np.log(sse / n) + p * np.log(n)

    dropped: str | float
    if dummy_family == "f_dummy":
        dropped = "F_0"
    elif dummy_family == "weekly":
        dropped = "W_1"
    else:
        dropped = np.nan

    return {
        "model_type": model_type,
        "dummy_family": dummy_family,
        "dropped_dummy": dropped,
        "train_rows": n,
        "test_rows": len(y_te),
        "predictors": str(features),
        "scaler_type": scaler_type,
        "n_observations": n,
        "n_test_observations": len(y_te),
        "R2": round(r2, 6),
        "AdjR2": round(adj_r2, 6),
        "MAE": round(mae, 6),
        "MAPE": round(mape, 6),
        "RMSE": round(rmse, 6),
        "Test_R2": round(test_r2, 6),
        "AIC": round(aic, 6),
        "BIC": round(bic, 6),
        "y_train_actual": y_tr,
        "y_train_pred": yhat_tr,
        "y_test_actual": y_te,
        "y_test_pred": yhat_te,
        "train_periods": train[["ISO_YEAR", "ISO_WEEK"]].values,
        "test_periods": test[["ISO_YEAR", "ISO_WEEK"]].values,
        "coefs": coefs,
        "features": features,
    }


def run_all_configs_for_entity(
    df: pd.DataFrame,
    scope: str,
    entity: str,
    channel: str,
) -> pd.DataFrame:
    """Run all 6 model configurations for one entity. Returns diagnostics rows."""
    entity_df = _resolve_entity_df(df, scope, entity)
    entity_df = entity_df.sort_values(["ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)

    display_cols = [
        "model_type", "dummy_family", "dropped_dummy", "train_rows", "test_rows",
        "predictors", "scaler_type", "n_observations", "n_test_observations",
        "R2", "AdjR2", "MAE", "MAPE", "RMSE", "Test_R2", "AIC", "BIC",
    ]

    rows = []
    for model_type in ["NNLS", "OLS"]:
        for dummy_family in ["f_dummy", "weekly", "fourier"]:
            result = fit_model_config(entity_df, channel, model_type, dummy_family)
            if result is not None:
                row = {
                    "scope": scope,
                    "entity": entity,
                    "channel": channel,
                    **{k: v for k, v in result.items() if k in display_cols},
                    "_result": result,
                }
                rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Version 6 — 8-iteration pipeline (OLS only, Prescreen variants)
# ---------------------------------------------------------------------------

V6_ITERATIONS = [
    {
        "num": 1,
        "label": "Week level columns",
        "dummy_family": "weekly",
        "prescreen_transform": None,
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 2,
        "label": "Bi-weekly level columns",
        "dummy_family": "f_dummy",
        "prescreen_transform": None,
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 3,
        "label": "Bi-weekly columns + Pre-screen spend split into (50 25, 25) across weeks 0, 1, 2",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag", "weights": [0.50, 0.25, 0.25], "lags": [0, 1, 2]},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 4,
        "label": "Bi-weekly columns + Pre-screen spend split into (25, 50, 25) across weeks -1, 0, 1",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag", "weights": [0.25, 0.50, 0.25], "lags": [-1, 0, 1]},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 5,
        "label": "Bi-weekly columns + Pre-screen spend split into (25, 50, 25) across weeks -1, 0, 1 + SQRT transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_sqrt", "weights": [0.25, 0.50, 0.25], "lags": [-1, 0, 1]},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 6,
        "label": "Bi-weekly columns + Pre-screen spend split into (25, 50, 25) across weeks -1, 0, 1 + LOG transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_log", "weights": [0.25, 0.50, 0.25], "lags": [-1, 0, 1]},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 7,
        "label": "Bi-weekly columns + Pre-screen spend split into (25, 50, 25) across weeks -1, 0, 1 + LOG transformation on Prescreen + Interaction effects Prescreen and DSP",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_log", "weights": [0.25, 0.50, 0.25], "lags": [-1, 0, 1]},
        "add_interaction": True,
        "drop_prescreen": False,
    },
    {
        "num": 8,
        "label": "Bi-weekly columns + Pre-screen spend split into (25, 50, 25) across weeks -1, 0, 1 + LOG transformation on Prescreen + Interaction effects Prescreen and DSP + Minus Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_log", "weights": [0.25, 0.50, 0.25], "lags": [-1, 0, 1]},
        "add_interaction": True,
        "drop_prescreen": True,
    },
    # --- Additional lag weight variants (all with LOG on Prescreen) ---
    {
        "num": 9,
        "label": "Bi-weekly columns + Pre-screen equal split (33, 34, 33) across weeks -1, 0, 1 + LOG transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_log", "weights": [0.33, 0.34, 0.33], "lags": [-1, 0, 1]},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 10,
        "label": "Bi-weekly columns + Pre-screen front-weighted (75, 25, 0) across weeks 0, 1, 2 + LOG transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_log", "weights": [0.75, 0.25, 0.0], "lags": [0, 1, 2]},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 11,
        "label": "Bi-weekly columns + Pre-screen delayed response (0, 25, 75) across weeks -1, 0, 1 + LOG transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_log", "weights": [0.0, 0.25, 0.75], "lags": [-1, 0, 1]},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 12,
        "label": "Bi-weekly columns + Pre-screen 4-week spread (10, 40, 40, 10) across weeks -1, 0, 1, 2 + LOG transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_log", "weights": [0.10, 0.40, 0.40, 0.10], "lags": [-1, 0, 1, 2]},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    # --- Geometric adstock variants (carryover effect on Prescreen) ---
    {
        "num": 13,
        "label": "Bi-weekly columns + Adstock α=0.3 on Prescreen (fast decay) + LOG transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "adstock_log", "alpha": 0.3},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 14,
        "label": "Bi-weekly columns + Adstock α=0.5 on Prescreen (medium decay) + LOG transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "adstock_log", "alpha": 0.5},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    {
        "num": 15,
        "label": "Bi-weekly columns + Adstock α=0.7 on Prescreen (slow decay) + LOG transformation on Prescreen",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "adstock_log", "alpha": 0.7},
        "add_interaction": False,
        "drop_prescreen": False,
    },
    # --- Saturation on both Prescreen and DSP ---
    {
        "num": 16,
        "label": "Bi-weekly columns + Pre-screen split (25, 50, 25) -1, 0, 1 + LOG Prescreen + LOG DSP (saturation on both key tactics)",
        "dummy_family": "f_dummy",
        "prescreen_transform": {"type": "lag_log", "weights": [0.25, 0.50, 0.25], "lags": [-1, 0, 1]},
        "add_interaction": False,
        "drop_prescreen": False,
        "log_tactics": ["DSP"],
    },
]


def _apply_prescreen_transform(entity_df: pd.DataFrame, transform: dict | None) -> pd.DataFrame:
    """Apply Prescreen transformation in-place on a copy. Lag is applied per STATE_CD."""
    if transform is None or "Prescreen" not in entity_df.columns:
        return entity_df.copy()

    df = entity_df.copy()
    t = transform["type"]

    if t == "sqrt":
        df["Prescreen"] = np.sqrt(df["Prescreen"].clip(lower=0))

    elif t == "log":
        df["Prescreen"] = np.log1p(df["Prescreen"].clip(lower=0))

    elif t == "lag":
        weights = transform["weights"]
        lags = transform["lags"]

        def _lag_group(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.sort_values(["ISO_YEAR", "ISO_WEEK"]).copy()
            ps = grp["Prescreen"].values.astype(float)
            n = len(ps)
            result = np.zeros(n)
            for w, lag in zip(weights, lags):
                if lag == 0:
                    result += w * ps
                elif lag > 0:               # lookback: result[t] uses ps[t-lag]
                    shifted = np.zeros(n)
                    shifted[lag:] = ps[: n - lag]
                    result += w * shifted
                else:                       # lookahead: result[t] uses ps[t+|lag|]
                    k = -lag
                    shifted = np.zeros(n)
                    shifted[: n - k] = ps[k:]
                    result += w * shifted
            grp["Prescreen"] = result
            return grp

        if "STATE_CD" in df.columns:
            df = df.groupby("STATE_CD", group_keys=False).apply(_lag_group)
        else:
            df = _lag_group(df)

    elif t in ("lag_sqrt", "lag_log"):
        # Apply lag first, then nonlinear transform
        lag_tr = {"type": "lag", "weights": transform["weights"], "lags": transform["lags"]}
        df = _apply_prescreen_transform(df, lag_tr)
        if t == "lag_sqrt":
            df["Prescreen"] = np.sqrt(df["Prescreen"].clip(lower=0))
        else:
            df["Prescreen"] = np.log1p(df["Prescreen"].clip(lower=0))

    elif t in ("adstock", "adstock_log"):
        # Geometric adstock: adstock[t] = spend[t] + alpha * adstock[t-1]
        alpha = float(transform.get("alpha", 0.5))

        def _adstock_group(grp: pd.DataFrame) -> pd.DataFrame:
            grp = grp.sort_values(["ISO_YEAR", "ISO_WEEK"]).copy()
            ps = grp["Prescreen"].values.astype(float)
            result = np.zeros(len(ps))
            result[0] = ps[0]
            for i in range(1, len(ps)):
                result[i] = ps[i] + alpha * result[i - 1]
            grp["Prescreen"] = result
            return grp

        if "STATE_CD" in df.columns:
            df = df.groupby("STATE_CD", group_keys=False).apply(_adstock_group)
        else:
            df = _adstock_group(df)
        if t == "adstock_log":
            df["Prescreen"] = np.log1p(df["Prescreen"].clip(lower=0))

    return df


def fit_v6_iteration(
    entity_df: pd.DataFrame,
    channel: str,
    dummy_family: str,
    prescreen_transform: dict | None = None,
    add_interaction: bool = False,
    drop_prescreen: bool = False,
    log_tactics: list | None = None,
    train_years: list | None = None,
    n_test_weeks: int = 8,
) -> dict | None:
    """
    Fit one V6 OLS iteration.
    Applies optional Prescreen transform (lag/sqrt/log/adstock) and optional
    Prescreen×DSP interaction term before fitting OLS (no intercept, MinMax
    scaler on tactic columns only). log_tactics applies log1p to named tactic
    columns before scaling (e.g. ["DSP"] for saturation on DSP spend).
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    if train_years is None:
        train_years = [2024, 2025]

    df = _apply_prescreen_transform(entity_df, prescreen_transform)
    df = df.sort_values(["ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)

    # Apply LOG to specified tactic columns before scaling (saturation transform)
    if log_tactics:
        for _col in log_tactics:
            if _col in df.columns:
                df[_col] = np.log1p(df[_col].clip(lower=0))

    tactics = [c for c in TACTIC_COLS_V5 if c in df.columns]
    if drop_prescreen:
        tactics = [t for t in tactics if t != "Prescreen"]

    family = DUMMY_FAMILIES_V5[dummy_family]
    seasonal = [f for f in family["features"]() if f in df.columns]
    if not tactics or not seasonal:
        return None

    train = df[df["ISO_YEAR"].isin(train_years)].copy()
    next_years = sorted([y for y in df["ISO_YEAR"].unique() if y not in train_years])
    if not next_years:
        return None
    test = (
        df[(df["ISO_YEAR"] == next_years[0]) & (df["ISO_WEEK"] <= n_test_weeks)]
        .sort_values(["ISO_YEAR", "ISO_WEEK"])
        .copy()
    )
    if len(train) < 10 or len(test) == 0:
        return None

    # Scale tactic columns (MinMax, fit on train only)
    tactic_scaler = MinMaxScaler()
    X_tr_tactic = tactic_scaler.fit_transform(train[tactics].values.astype(float))
    X_te_tactic = tactic_scaler.transform(test[tactics].values.astype(float))

    X_tr_seasonal = train[seasonal].values.astype(float)
    X_te_seasonal = test[seasonal].values.astype(float)

    # Optional Prescreen×DSP interaction (computed on scaled values)
    extra_features: list[str] = []
    X_tr_extra = np.empty((len(train), 0))
    X_te_extra = np.empty((len(test), 0))

    if add_interaction and "DSP" in tactics:
        dsp_idx = tactics.index("DSP")
        if "Prescreen" in tactics:
            ps_idx = tactics.index("Prescreen")
            inter_tr = (X_tr_tactic[:, ps_idx] * X_tr_tactic[:, dsp_idx]).reshape(-1, 1)
            inter_te = (X_te_tactic[:, ps_idx] * X_te_tactic[:, dsp_idx]).reshape(-1, 1)
        elif "Prescreen" in df.columns:
            # Prescreen was dropped from tactics; scale it separately for the interaction
            ps_scaler = MinMaxScaler()
            ps_tr = ps_scaler.fit_transform(train[["Prescreen"]].values.astype(float))
            ps_te = ps_scaler.transform(test[["Prescreen"]].values.astype(float))
            inter_tr = (ps_tr[:, 0] * X_tr_tactic[:, dsp_idx]).reshape(-1, 1)
            inter_te = (ps_te[:, 0] * X_te_tactic[:, dsp_idx]).reshape(-1, 1)
        else:
            inter_tr = inter_te = None

        if inter_tr is not None:
            X_tr_extra = inter_tr
            X_te_extra = inter_te
            extra_features = ["Prescreen_x_DSP"]

    X_tr = np.hstack([X_tr_tactic, X_tr_seasonal, X_tr_extra])
    X_te = np.hstack([X_te_tactic, X_te_seasonal, X_te_extra])
    all_features = tactics + seasonal + extra_features

    y_tr = train["NON_DM_APPLICATIONS"].values.astype(float)
    y_te = test["NON_DM_APPLICATIONS"].values.astype(float)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_tr, y_tr)
    yhat_tr = lr.predict(X_tr)
    yhat_te = lr.predict(X_te)
    coefs = lr.coef_

    n, p = len(y_tr), len(all_features)
    r2 = float(r2_score(y_tr, yhat_tr))
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)
    mae = float(mean_absolute_error(y_tr, yhat_tr))
    mask = y_tr != 0
    mape = float(np.mean(np.abs((y_tr[mask] - yhat_tr[mask]) / y_tr[mask])) * 100) if mask.any() else np.nan
    rmse = float(np.sqrt(mean_squared_error(y_tr, yhat_tr)))
    test_r2 = float(r2_score(y_te, yhat_te)) if len(y_te) > 0 else np.nan
    sse = float(np.sum((y_tr - yhat_tr) ** 2))
    aic = float(n * np.log(sse / n) + 2 * p)
    bic = float(n * np.log(sse / n) + p * np.log(n))

    return {
        "model_type": "OLS",
        "dummy_family": dummy_family,
        "train_rows": n,
        "test_rows": len(y_te),
        "R2": round(r2, 6),
        "AdjR2": round(adj_r2, 6),
        "MAE": round(mae, 6),
        "MAPE": round(mape, 6),
        "RMSE": round(rmse, 6),
        "Test_R2": round(test_r2, 6),
        "AIC": round(aic, 6),
        "BIC": round(bic, 6),
        "features": all_features,
        "tactic_cols": tactics,
        "seasonal_cols": seasonal,
        "coefs": coefs,
        "scaler": tactic_scaler,
        "y_train_actual": y_tr,
        "y_train_pred": yhat_tr,
        "y_test_actual": y_te,
        "y_test_pred": yhat_te,
        "train_periods": train[["ISO_YEAR", "ISO_WEEK"]].values,
        "test_periods": test[["ISO_YEAR", "ISO_WEEK"]].values,
        "X_train_scaled": X_tr,
    }


def run_v6_iterations_for_entity(
    df: pd.DataFrame,
    scope: str,
    entity: str,
    channel: str,
) -> pd.DataFrame:
    """Run all 8 V6 iterations (OLS) for one entity. Returns diagnostics DataFrame."""
    entity_df = _resolve_entity_df(df, scope, entity)
    entity_df = entity_df.sort_values(["ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)

    diag_keys = [
        "model_type", "dummy_family", "train_rows", "test_rows",
        "R2", "AdjR2", "MAE", "MAPE", "RMSE", "Test_R2", "AIC", "BIC",
    ]
    rows = []
    for cfg in V6_ITERATIONS:
        result = fit_v6_iteration(
            entity_df, channel,
            dummy_family=cfg["dummy_family"],
            prescreen_transform=cfg["prescreen_transform"],
            add_interaction=cfg["add_interaction"],
            drop_prescreen=cfg["drop_prescreen"],
            log_tactics=cfg.get("log_tactics"),
        )
        if result is not None:
            rows.append({
                "iteration": cfg["num"],
                "label": cfg["label"],
                "scope": scope,
                "entity": entity,
                "channel": channel,
                **{k: v for k, v in result.items() if k in diag_keys},
                "_result": result,
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _resolve_entity_df(df: pd.DataFrame, scope: str, entity: str) -> pd.DataFrame:
    """Return a single-entity DataFrame, summing tactic/outcome cols for division scope."""
    if scope == "state":
        return df[df["STATE_CD"] == entity].copy()
    div_df = df[df["Division"] == entity].copy()
    tactic_sum = [c for c in TACTIC_COLS_V5 if c in div_df.columns]
    outcome_sum = [c for c in ["APPLICATIONS", "NON_DM_APPLICATIONS"] if c in div_df.columns]
    seasonal_first = [c for c in div_df.columns if c.startswith(("F_", "W_", "sin_", "cos_"))]
    agg: dict = {c: "sum" for c in tactic_sum + outcome_sum}
    agg.update({c: "first" for c in seasonal_first})
    return div_df.groupby(["ISO_YEAR", "ISO_WEEK"]).agg(agg).reset_index()


def run_ols_configs_for_entity(
    df: pd.DataFrame,
    scope: str,
    entity: str,
    channel: str,
) -> pd.DataFrame:
    """
    Version 6 pipeline: OLS only, weekly and fortnightly dummies only (no NNLS, no Fourier).
    Returns 2 diagnostic rows: OLS|weekly and OLS|f_dummy.
    """
    entity_df = _resolve_entity_df(df, scope, entity)
    entity_df = entity_df.sort_values(["ISO_YEAR", "ISO_WEEK"]).reset_index(drop=True)

    display_cols = [
        "model_type", "dummy_family", "dropped_dummy", "train_rows", "test_rows",
        "predictors", "scaler_type", "n_observations", "n_test_observations",
        "R2", "AdjR2", "MAE", "MAPE", "RMSE", "Test_R2", "AIC", "BIC",
    ]

    rows = []
    for dummy_family in ["weekly", "f_dummy"]:
        result = fit_model_config(entity_df, channel, "OLS", dummy_family)
        if result is not None:
            row = {
                "scope": scope,
                "entity": entity,
                "channel": channel,
                **{k: v for k, v in result.items() if k in display_cols},
                "_result": result,
            }
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
