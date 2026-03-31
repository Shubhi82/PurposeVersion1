from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CANDIDATES = [
    ROOT / "All_States_modeling_v2.xlsx",
    ROOT / "All_States_modeling.xlsx",
    Path.home() / "Downloads" / "All_States_modeling.xlsx",
]
DEFAULT_DATA_PATH = next((path for path in DEFAULT_DATA_CANDIDATES if path.exists()), DEFAULT_DATA_CANDIDATES[0])
MARKETING_SPEND_CANDIDATES = [
    ROOT / "Marketing Spend Data.csv",
    ROOT / "Marketing_Spend_Data.csv",
    ROOT / "Marketing Spend Data (1).csv",
]
ORIGINATIONS_CANDIDATES = [
    ROOT / "Originations Data1 (1).xlsx",
    ROOT / "Originations Data1.xlsx",
    ROOT / "Originations Data.xlsx",
    ROOT / "Originations_Data.xlsx",
    ROOT / "Originations_Data.csv",
]
DM_DATA_CANDIDATES = [
    ROOT / "DM Data.csv",
    ROOT / "DM Data .csv",
    Path.home() / "Downloads" / "DM Data.csv",
    Path.home() / "Downloads" / "DM Data .csv",
]
MARKETING_SPEND_PATH = next(
    (path for path in MARKETING_SPEND_CANDIDATES if path.exists()),
    MARKETING_SPEND_CANDIDATES[0],
)
ORIGINATIONS_PATH = next(
    (path for path in ORIGINATIONS_CANDIDATES if path.exists()),
    ORIGINATIONS_CANDIDATES[0],
)
DM_DATA_PATH = next(
    (path for path in DM_DATA_CANDIDATES if path.exists()),
    DM_DATA_CANDIDATES[0],
)

TACTIC_COLUMNS = [
    "DSP",
    "LeadGen",
    "Paid Search",
    "Paid Social",
    "Prescreen",
    "Referrals",
    "Sweepstakes",
]

OUTCOME_COLUMNS = ["APPLICATIONS", "APPROVED", "ORIGINATIONS"]

CHANNELS = ("DIGITAL", "PHYSICAL")
PRODUCT_ALL_LABEL = "All Products"
TIME_GRAINS = ("Weekly", "Fortnight")


def iso_week_start(iso_year: int, iso_week: int) -> pd.Timestamp:
    return pd.Timestamp(date.fromisocalendar(int(iso_year), int(iso_week), 1))


def add_time_columns(df: pd.DataFrame, time_grain: str) -> pd.DataFrame:
    enriched = df.copy()
    enriched["ISO_YEAR"] = enriched["ISO_YEAR"].astype(int)
    enriched["ISO_WEEK"] = enriched["ISO_WEEK"].astype(int)
    enriched["period_start"] = [
        iso_week_start(year, week)
        for year, week in zip(enriched["ISO_YEAR"], enriched["ISO_WEEK"])
    ]

    if time_grain == "Weekly":
        enriched["period_number"] = enriched["ISO_WEEK"]
        enriched["period_id"] = (
            enriched["ISO_YEAR"].astype(str)
            + "-W"
            + enriched["ISO_WEEK"].astype(str).str.zfill(2)
        )
        enriched["period_label"] = enriched["period_id"]
        return enriched

    enriched["period_number"] = ((enriched["ISO_WEEK"] - 1) // 2) + 1
    enriched["fortnight_start_week"] = ((enriched["period_number"] - 1) * 2) + 1
    enriched["period_start"] = [
        iso_week_start(year, week)
        for year, week in zip(enriched["ISO_YEAR"], enriched["fortnight_start_week"])
    ]
    enriched["period_id"] = (
        enriched["ISO_YEAR"].astype(str)
        + "-F"
        + enriched["period_number"].astype(str).str.zfill(2)
    )
    enriched["period_label"] = enriched["period_id"]
    return enriched


def get_available_products(df: pd.DataFrame, state: str) -> list[str]:
    products = (
        df.loc[df["STATE_CD"] == state, "PRODUCT_CD"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )
    return [PRODUCT_ALL_LABEL, *products]


def is_rolled_up_product(product: str | None) -> bool:
    value = "" if product is None else str(product).strip()
    return "/" in value


def get_available_rolled_up_products(df: pd.DataFrame, state: str) -> list[str]:
    products = (
        df.loc[df["STATE_CD"] == state, "PRODUCT_CD"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    rolled_up = sorted(products[products.apply(is_rolled_up_product)].unique().tolist())
    return [PRODUCT_ALL_LABEL, *rolled_up]


def expand_rollup_product(product: str | None) -> list[str]:
    if not product or product == PRODUCT_ALL_LABEL:
        return []
    parts = [part.strip() for part in str(product).split("/") if part.strip()]
    return parts if parts else [str(product).strip()]


def filter_data(
    df: pd.DataFrame,
    state: str | None = None,
    product: str | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if state:
        filtered = filtered.loc[filtered["STATE_CD"] == state]
    if product and product != PRODUCT_ALL_LABEL:
        filtered = filtered.loc[filtered["PRODUCT_CD"] == product]
    return filtered


def compute_mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    mask = actual_array != 0
    if not mask.any():
        return float("nan")
    return float(
        np.mean(np.abs((actual_array[mask] - predicted_array[mask]) / actual_array[mask])) * 100
    )


def format_metric(value: float | int | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:,.{decimals}f}"


def make_interpretation_sentence(channel: str, tactic: str, coefficient: float) -> str:
    direction = "increases" if coefficient >= 0 else "decreases"
    return (
        f"1 unit increase in {channel} {tactic} spend {direction} "
        f"{channel} applications by {abs(coefficient):.2f}, holding the other tactics "
        "and time dummies constant."
    )
