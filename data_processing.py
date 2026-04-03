from __future__ import annotations

from pathlib import Path
import pandas as pd

# =========================
# PATH CONFIG
# =========================
ROOT = Path(__file__).resolve().parent

ORIGINATIONS_PATH = ROOT / "Originations_Data.csv"   # or .xlsx
DM_DIFF_PATH      = ROOT / "DM_Negative_Differences.xlsx"


# =========================
# LOAD DATA
# =========================
def load_originations_data(path: Path = ORIGINATIONS_PATH) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df.columns = df.columns.str.upper()
    return df


def load_dm_differences(path: Path = DM_DIFF_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.upper()
    return df


# =========================
# BUILD NON-DM TARGET
# =========================
def build_non_dm_application_series() -> pd.DataFrame:
    """
    Version 7 Logic:
    NON_DM_APPLICATIONS = APPLICATIONS - DM_APPLICATIONS
    """

    originations = load_originations_data()
    dm_diff = load_dm_differences()

    # -------------------------
    # Aggregate originations
    # -------------------------
    orig_agg = (
        originations
        .groupby(["ISO_YEAR", "ISO_WEEK", "STATE_CD"], as_index=False)
        ["APPLICATIONS"]
        .sum()
    )

    # -------------------------
    # Clean DM differences
    # -------------------------
    dm_clean = (
        dm_diff
        .groupby(["ISO_YEAR", "ISO_WEEK", "STATE_CD"], as_index=False)
        ["DM_APPLICATIONS"]
        .sum()
    )

    # -------------------------
    # Merge
    # -------------------------
    df = orig_agg.merge(
        dm_clean,
        on=["ISO_YEAR", "ISO_WEEK", "STATE_CD"],
        how="left"
    )

    # -------------------------
    # Fill missing DM
    # -------------------------
    df["DM_APPLICATIONS"] = df["DM_APPLICATIONS"].fillna(0)

    # -------------------------
    # 🔥 CORE STEP
    # -------------------------
    df["NON_DM_APPLICATIONS"] = (
        df["APPLICATIONS"] - df["DM_APPLICATIONS"]
    )

    # -------------------------
    # Safety check
    # -------------------------
    df["NON_DM_APPLICATIONS"] = df["NON_DM_APPLICATIONS"].clip(lower=0)

    return df


# =========================
# FINAL MODEL DATASET
# =========================
def get_modeling_dataset(marketing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines NON-DM target with marketing features
    """

    target_df = build_non_dm_application_series()

    # Merge with marketing spend data
    df = target_df.merge(
        marketing_df,
        on=["ISO_YEAR", "ISO_WEEK", "STATE_CD"],
        how="left"
    )

    # Final target
    df["TARGET"] = df["NON_DM_APPLICATIONS"]

    return df
