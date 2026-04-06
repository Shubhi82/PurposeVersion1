from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import nnls
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


TARGET_COL = "NON_DM_APPLICATIONS"
#TARGET_COL = "APPLICATIONS"
STATE_COL = "STATE_CD"
DIVISION_COL = "Division"
YEAR_COL = "ISO_YEAR"
WEEK_COL = "ISO_WEEK"

NON_DUMMY_PREDICTORS = [
    "DSP",
    "LeadGen",
    "Paid Search",
    "Paid Social",
    "Prescreen",
    "Referrals",
]
DEFAULT_MEDIA_PREDICTORS = list(NON_DUMMY_PREDICTORS)

DUMMY_FAMILIES: Dict[str, List[str]] = {
    "weekly": [f"W_{idx}" for idx in range(1, 53)],
    "f_dummy": [f"F_{idx}" for idx in range(26)],
}
FOURIER_COLS = ["sin_1", "cos_1", "sin_2", "cos_2"]
FEATURE_RUNS: Dict[str, Dict[str, object]] = {
    "weekly": {
        "extra_cols": DUMMY_FAMILIES["weekly"],
        "drop_one": True,
        "scaler": "minmax",
    },
    "f_dummy": {
        "extra_cols": DUMMY_FAMILIES["f_dummy"],
        "drop_one": True,
        "scaler": "minmax",
    },
    "fourier": {
        "extra_cols": FOURIER_COLS,
        "drop_one": False,
        "scaler": "standard",
    },
}

TRAIN_YEARS = {2024, 2025}
TEST_YEAR = 2026
TEST_WEEKS = set(range(1, 9))
EPSILON = 1e-9
BACKTEST_MODES = {"fixed_holdout", "rolling_one_step_expanding", "rolling_one_step_fixed_window"}
DEFAULT_FIXED_WINDOW_WEEKS = 104

TIME_INDEX_COL = "time_index"
TIME_INDEX_SQ_COL = "time_index_sq"
PRESCREEN_LAG1_COL = "Prescreen_lag1"
DSP_LAG1_COL = "DSP_lag1"
PAID_SEARCH_LAG1_COL = "Paid_Search_lag1"
DSP_TRAILING_4W_AVG_COL = "DSP_trailing_4w_avg"
PAID_SEARCH_TRAILING_4W_AVG_COL = "Paid_Search_trailing_4w_avg"
PRESCREEN_TRAILING_4W_AVG_COL = "Prescreen_trailing_4w_avg"
YEAR_INDICATOR_2025_COL = "year_indicator_2025"
YEAR_INDICATOR_2026_COL = "year_indicator_2026"
TARGET_LAG1_COL = f"{TARGET_COL}_lag1"
TARGET_TRAILING_4W_AVG_COL = f"{TARGET_COL}_trailing_4w_avg"

OPTIONAL_FEATURES: Dict[str, str] = {
    TIME_INDEX_COL: "Sequential week counter within each state or aggregated division series.",
    TIME_INDEX_SQ_COL: "Squared weekly trend term to capture curved long-run growth or decline.",
    YEAR_INDICATOR_2025_COL: "Indicator equal to 1 for 2025 rows and 0 otherwise.",
    YEAR_INDICATOR_2026_COL: "Indicator equal to 1 for 2026 rows and 0 otherwise.",
    PRESCREEN_LAG1_COL: "Prior week's Prescreen volume.",
    DSP_LAG1_COL: "Prior week's DSP volume.",
    PAID_SEARCH_LAG1_COL: "Prior week's Paid Search volume.",
    DSP_TRAILING_4W_AVG_COL: "Rolling 4-week average of DSP, inclusive of the current week.",
    PAID_SEARCH_TRAILING_4W_AVG_COL: "Rolling 4-week average of Paid Search, inclusive of the current week.",
    PRESCREEN_TRAILING_4W_AVG_COL: "Rolling 4-week average of Prescreen, inclusive of the current week.",
    TARGET_LAG1_COL: "Prior week's observed target value.",
    TARGET_TRAILING_4W_AVG_COL: "Rolling 4-week average of the observed target using only prior weeks.",
}
SATURATION_METHODS = {"none", "log1p"}


@dataclass
class RunMetadata:
    """Metadata saved with each trained model artifact or shown during inline review.

    Attributes:
        scope: Either ``state`` or ``division``.
        entity: State code or division name being modeled.
        model_type: ``OLS`` or ``NNLS``.
        dummy_family: One of ``weekly``, ``f_dummy``, or ``fourier``.
        dropped_dummy: Baseline dummy removed to avoid collinearity when a dummy family is used.
        train_rows: Number of training observations.
        test_rows: Number of holdout observations.
        predictors: Final predictor list after combining the default variables, seasonal terms,
            and any user-selected optional engineered variables.
        scaler_type: ``minmax`` for the default seasonal runs and ``standard`` for the Fourier run.
        backtest_mode: Backtest style used for evaluation.
        media_transform_config: Mapping of raw media variables to the transform settings used.
    """

    scope: str
    entity: str
    model_type: str
    dummy_family: str
    dropped_dummy: Optional[str]
    train_rows: int
    test_rows: int
    predictors: List[str]
    scaler_type: str
    backtest_mode: str
    media_transform_config: Dict[str, Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    The command-line interface keeps the current file-writing behavior as the default.
    For notebook usage, prefer importing :func:`run_model_pipeline` and passing keyword
    arguments directly instead of going through the CLI.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build OLS and NNLS models for each State and Division using 2024-2025 data, "
            "then test on 2026 weeks 1-8."
        )
    )
    parser.add_argument(
        "--input",
        default="/Users/Rahul/Desktop/Code/Working Codebase/ModelingFile_Digital.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="state_division_model_artifacts",
        help="Directory where diagnostics, models, scalers, and coefficient outputs will be saved.",
    )
    parser.add_argument(
        "--inline-output",
        action="store_true",
        help="Display inline notebook-style output instead of writing files. Requires selected states/divisions.",
    )
    parser.add_argument(
        "--selected-states",
        nargs="*",
        default=None,
        help="Optional list of state codes to run.",
    )
    parser.add_argument(
        "--selected-divisions",
        nargs="*",
        default=None,
        help="Optional list of division names to run.",
    )
    parser.add_argument(
        "--methodologies",
        nargs="*",
        default=None,
        help="Optional list from: OLS, NNLS, weekly, f_dummy, Fourier.",
    )
    parser.add_argument(
        "--optional-features",
        nargs="*",
        default=None,
        help=f"Optional engineered variables to add. Choices: {', '.join(sorted(OPTIONAL_FEATURES))}.",
    )
    parser.add_argument(
        "--media-predictors",
        nargs="*",
        default=None,
        help=f"Optional subset of base media predictors. Choices: {', '.join(NON_DUMMY_PREDICTORS)}.",
    )
    parser.add_argument(
        "--backtest-mode",
        default="fixed_holdout",
        choices=sorted(BACKTEST_MODES),
        help="Backtest style: fixed_holdout, rolling_one_step_expanding, or rolling_one_step_fixed_window.",
    )
    parser.add_argument(
        "--fixed-window-weeks",
        type=int,
        default=DEFAULT_FIXED_WINDOW_WEEKS,
        help="Training window length for rolling_one_step_fixed_window.",
    )
    return parser.parse_args()


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def load_data(input_path: str) -> pd.DataFrame:
    """Load and validate the modeling dataset.

    Required columns include the target, the six default non-dummy marketing variables,
    the weekly dummy family, the ``F_*`` dummy family, and the four Fourier columns
    ``sin_1``, ``cos_1``, ``sin_2``, ``cos_2``.
    """
    df = pd.read_csv(input_path)

    required_cols = {
        YEAR_COL,
        WEEK_COL,
        STATE_COL,
        DIVISION_COL,
        TARGET_COL,
        *NON_DUMMY_PREDICTORS,
        *(col for cols in DUMMY_FAMILIES.values() for col in cols),
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    missing_fourier = [col for col in FOURIER_COLS if col not in df.columns]
    if missing_fourier:
        raise ValueError(f"Missing required Fourier columns: {missing_fourier}")

    df = df.copy()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[WEEK_COL] = pd.to_numeric(df[WEEK_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    numeric_cols = list(NON_DUMMY_PREDICTORS)
    numeric_cols.extend(col for cols in DUMMY_FAMILIES.values() for col in cols)
    numeric_cols.extend(FOURIER_COLS)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df.dropna(subset=[YEAR_COL, WEEK_COL, TARGET_COL, STATE_COL, DIVISION_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df[WEEK_COL] = df[WEEK_COL].astype(int)
    df[STATE_COL] = df[STATE_COL].astype(str)
    df[DIVISION_COL] = df[DIVISION_COL].astype(str)

    return df.sort_values([YEAR_COL, WEEK_COL, STATE_COL, DIVISION_COL]).reset_index(drop=True)


def validate_optional_features(optional_features: Optional[Sequence[str]]) -> List[str]:
    """Validate requested engineered variables.

    Available optional variables:
        - ``time_index``
        - ``time_index_sq``
        - ``year_indicator_2025``
        - ``year_indicator_2026``
        - ``Prescreen_lag1``
        - ``DSP_lag1``
        - ``Paid_Search_lag1``
        - ``DSP_trailing_4w_avg``
        - ``Paid_Search_trailing_4w_avg``
        - ``Prescreen_trailing_4w_avg``
        - ``APPLICATIONS_lag1`` or ``NON_DM_APPLICATIONS_lag1`` depending on the active target
        - ``APPLICATIONS_trailing_4w_avg`` or ``NON_DM_APPLICATIONS_trailing_4w_avg`` depending on the active target
    """
    if optional_features is None:
        return []

    requested = []
    for feature in optional_features:
        if feature not in OPTIONAL_FEATURES:
            raise ValueError(
                f"Unsupported optional feature '{feature}'. Available options: {sorted(OPTIONAL_FEATURES)}"
            )
        requested.append(feature)
    return requested


def validate_media_predictors(media_predictors: Optional[Sequence[str]]) -> List[str]:
    """Validate the selected base media predictors.

    If ``media_predictors`` is omitted, the script uses all six default media variables:
        - ``DSP``
        - ``LeadGen``
        - ``Paid Search``
        - ``Paid Social``
        - ``Prescreen``
        - ``Referrals``
    """
    if media_predictors is None:
        return list(DEFAULT_MEDIA_PREDICTORS)

    requested = []
    for predictor in media_predictors:
        if predictor not in NON_DUMMY_PREDICTORS:
            raise ValueError(
                f"Unsupported media predictor '{predictor}'. Available options: {NON_DUMMY_PREDICTORS}"
            )
        requested.append(predictor)

    if not requested:
        raise ValueError("At least one media predictor must be selected.")

    return requested


def validate_media_transform_config(
    media_transform_config: Optional[Dict[str, Dict[str, Any]]],
    media_predictors: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Validate optional media transform settings.

    Design rules:
        - If a selected media variable is not listed here, it is used raw.
        - If a selected media variable is listed here, only the transformed variant is used.
        - Raw and transformed forms of the same variable are not both included in the model.

    Supported per-variable keys:
        - ``alpha``: adstock carryover value between 0 and 1
        - ``saturation``: ``none`` or ``log1p``
    """
    if media_transform_config is None:
        return {}

    validated: Dict[str, Dict[str, Any]] = {}
    for media_name, raw_config in media_transform_config.items():
        if media_name not in media_predictors:
            raise ValueError(
                f"Transform config provided for '{media_name}', but it is not in selected media_predictors."
            )
        if not isinstance(raw_config, dict):
            raise ValueError(f"Transform config for '{media_name}' must be a dictionary.")

        alpha = raw_config.get("alpha", 0.0)
        saturation = str(raw_config.get("saturation", "none")).lower()

        if alpha is None:
            alpha = 0.0
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha for '{media_name}' must be between 0 and 1.")
        if saturation not in SATURATION_METHODS:
            raise ValueError(
                f"Unsupported saturation method '{saturation}' for '{media_name}'. "
                f"Choices: {sorted(SATURATION_METHODS)}"
            )

        if alpha == 0.0 and saturation == "none":
            continue

        validated[media_name] = {
            "alpha": alpha,
            "saturation": saturation,
        }

    return validated


def parse_methodology_selection(methodologies: Optional[Sequence[str]]) -> Tuple[Set[str], Set[str]]:
    """Parse a user-friendly methodology selection list.

    Supported values:
        - ``OLS`` and ``NNLS`` filter model types
        - ``weekly``, ``f_dummy``, and ``fourier`` filter seasonal feature runs
        - ``Fourier`` is accepted as a friendly alias for the ``fourier`` feature run

    If no model type is supplied, both ``OLS`` and ``NNLS`` are used.
    If no feature run is supplied, all feature runs are used.
    """
    if methodologies is None:
        return {"OLS", "NNLS"}, set(FEATURE_RUNS.keys())

    model_types: Set[str] = set()
    feature_runs: Set[str] = set()
    aliases = {"fourier": "fourier", "weekly": "weekly", "f_dummy": "f_dummy"}

    for item in methodologies:
        token = str(item).strip()
        upper_token = token.upper()
        lower_token = token.lower()
        if upper_token in {"OLS", "NNLS"}:
            model_types.add(upper_token)
        elif lower_token in aliases:
            feature_runs.add(aliases[lower_token])
        else:
            raise ValueError(
                "Unsupported methodology selection "
                f"'{item}'. Use OLS, NNLS, weekly, f_dummy, or Fourier."
            )

    if not model_types:
        model_types = {"OLS", "NNLS"}
    if not feature_runs:
        feature_runs = set(FEATURE_RUNS.keys())

    return model_types, feature_runs


def prepare_entity_subset(
    df: pd.DataFrame,
    scope_col: str,
    selected_entities: Optional[Sequence[str]],
) -> pd.DataFrame:
    """Filter to a user-specified state or division subset when requested."""
    if not selected_entities:
        return df

    selected_lookup = {str(value) for value in selected_entities}
    return df[df[scope_col].astype(str).isin(selected_lookup)].copy()


def select_run_columns(df: pd.DataFrame, run_name: str) -> Tuple[List[str], Optional[str]]:
    run_config = FEATURE_RUNS[run_name]
    candidates = [col for col in run_config["extra_cols"] if col in df.columns]
    available = [col for col in candidates if df[col].fillna(0.0).abs().sum() > 0]

    if not available:
        return [], None

    if run_config["drop_one"]:
        dropped_col = available[0]
        selected = [col for col in available if col != dropped_col]
        return selected, dropped_col

    return available, None


def apply_recursive_adstock(series: pd.Series, alpha: float) -> pd.Series:
    """Apply simple recursive adstock to a media series."""
    values = series.fillna(0.0).astype(float).to_numpy()
    out = np.zeros(len(values), dtype=float)
    carry = 0.0
    for idx, value in enumerate(values):
        carry = value + alpha * carry
        out[idx] = carry
    return pd.Series(out, index=series.index, dtype=float)


def apply_saturation(series: pd.Series, saturation: str) -> pd.Series:
    """Apply the requested saturation transform."""
    values = series.fillna(0.0).astype(float)
    if saturation == "none":
        return values
    if saturation == "log1p":
        return np.log1p(np.clip(values, a_min=0.0, a_max=None))
    raise ValueError(f"Unsupported saturation method '{saturation}'.")


def transform_media_variables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    media_predictors: Sequence[str],
    media_transform_config: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    """Create media design columns, using raw or transformed variants channel by channel.

    Any selected media variable not listed in ``media_transform_config`` is used raw.
    Any selected media variable listed there is replaced by its transformed counterpart.
    """
    combined = pd.concat(
        [
            train_df.assign(__dataset="train"),
            test_df.assign(__dataset="test"),
        ],
        axis=0,
        ignore_index=True,
    )
    combined = combined.sort_values([YEAR_COL, WEEK_COL, "__dataset"]).reset_index(drop=True)

    selected_feature_names: List[str] = []
    feature_name_map: Dict[str, str] = {}

    for media_name in media_predictors:
        if media_name not in media_transform_config:
            selected_feature_names.append(media_name)
            feature_name_map[media_name] = media_name
            continue

        config = media_transform_config[media_name]
        alpha = float(config.get("alpha", 0.0))
        saturation = str(config.get("saturation", "none")).lower()

        transformed = combined[media_name].astype(float)
        feature_suffixes: List[str] = []
        if alpha > 0.0:
            transformed = apply_recursive_adstock(transformed, alpha)
            feature_suffixes.append(f"adstock_{alpha:g}")
        if saturation != "none":
            transformed = apply_saturation(transformed, saturation)
            feature_suffixes.append(saturation)

        transformed_name = f"{safe_name(media_name)}__{'__'.join(feature_suffixes)}"
        combined[transformed_name] = transformed.astype(float)
        selected_feature_names.append(transformed_name)
        feature_name_map[media_name] = transformed_name

    train_out = combined[combined["__dataset"] == "train"].drop(columns=["__dataset"]).copy()
    test_out = combined[combined["__dataset"] == "test"].drop(columns=["__dataset"]).copy()
    return train_out, test_out, selected_feature_names, feature_name_map


def split_train_test(entity_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = entity_df[entity_df[YEAR_COL].isin(TRAIN_YEARS)].copy()
    test_df = entity_df[
        (entity_df[YEAR_COL] == TEST_YEAR) & (entity_df[WEEK_COL].isin(TEST_WEEKS))
    ].copy()
    return train_df, test_df


def generate_backtest_splits(
    entity_df: pd.DataFrame,
    backtest_mode: str,
    fixed_window_weeks: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build train/test splits for the requested backtest mode.

    Modes:
        - ``fixed_holdout``: one model fit on 2024-2025, scored on 2026 weeks 1-8.
        - ``rolling_one_step_expanding``: refit each week using all prior observed rows.
        - ``rolling_one_step_fixed_window``: refit each week using only the most recent
          ``fixed_window_weeks`` prior rows.
    """
    if backtest_mode not in BACKTEST_MODES:
        raise ValueError(f"Unsupported backtest mode '{backtest_mode}'.")

    entity_df = entity_df.sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)
    if backtest_mode == "fixed_holdout":
        train_df, test_df = split_train_test(entity_df)
        return [(train_df, test_df)] if (not train_df.empty and not test_df.empty) else []

    candidate_idx = entity_df.index[
        (entity_df[YEAR_COL] == TEST_YEAR) & (entity_df[WEEK_COL].isin(TEST_WEEKS))
    ].tolist()
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for idx in candidate_idx:
        train_df = entity_df.iloc[:idx].copy()
        if backtest_mode == "rolling_one_step_fixed_window" and fixed_window_weeks > 0:
            train_df = train_df.tail(fixed_window_weeks).copy()
        test_df = entity_df.iloc[[idx]].copy()
        if not train_df.empty and not test_df.empty:
            splits.append((train_df, test_df))
    return splits


def aggregate_division_weekly(entity_df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    """Aggregate state rows to one row per division-week for division fallback models.

    Additive series are summed across component states, while seasonal columns are carried
    forward once per division-week because they describe the week itself rather than volume.
    All raw media columns are retained in the aggregated frame so optional lag and rolling
    features can still be engineered even when the user later models only a subset.
    """
    run_cols = [col for col in FEATURE_RUNS[run_name]["extra_cols"] if col in entity_df.columns]
    group_cols = [DIVISION_COL, YEAR_COL, WEEK_COL]
    additive_cols = [TARGET_COL] + list(NON_DUMMY_PREDICTORS)
    aggregation_map = {col: "sum" for col in additive_cols}
    aggregation_map.update({col: "first" for col in run_cols})

    aggregated = (
        entity_df[group_cols + additive_cols + run_cols]
        .groupby(group_cols, as_index=False)
        .agg(aggregation_map)
    )

    return aggregated.sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)


def add_optional_features(entity_df: pd.DataFrame, optional_features: Sequence[str]) -> pd.DataFrame:
    """Create engineered variables that users can optionally include in the model.

    Available optional variables:
        - ``time_index``
        - ``time_index_sq``
        - ``year_indicator_2025``
        - ``year_indicator_2026``
        - ``Prescreen_lag1``
        - ``DSP_lag1``
        - ``Paid_Search_lag1``
        - ``DSP_trailing_4w_avg``
        - ``Paid_Search_trailing_4w_avg``
        - ``Prescreen_trailing_4w_avg``
        - ``APPLICATIONS_lag1`` or ``NON_DM_APPLICATIONS_lag1`` depending on the active target
        - ``APPLICATIONS_trailing_4w_avg`` or ``NON_DM_APPLICATIONS_trailing_4w_avg`` depending on the active target

    The function expects data for a single modeling series, meaning either one state over
    time or one aggregated division over time. Lagged terms are filled with ``0.0`` for the
    earliest week so the design matrix stays dense. Target trailing averages use only prior
    observed weeks to avoid leaking the current target into the predictor set.
    """
    if not optional_features:
        return entity_df

    frame = entity_df.sort_values([YEAR_COL, WEEK_COL]).copy()
    time_index = np.arange(1, len(frame) + 1, dtype=float)
    lagged_target = frame[TARGET_COL].shift(1)

    if TIME_INDEX_COL in optional_features:
        frame[TIME_INDEX_COL] = time_index
    if TIME_INDEX_SQ_COL in optional_features:
        frame[TIME_INDEX_SQ_COL] = np.square(time_index)
    if YEAR_INDICATOR_2025_COL in optional_features:
        frame[YEAR_INDICATOR_2025_COL] = (frame[YEAR_COL] == 2025).astype(float)
    if YEAR_INDICATOR_2026_COL in optional_features:
        frame[YEAR_INDICATOR_2026_COL] = (frame[YEAR_COL] == 2026).astype(float)
    if PRESCREEN_LAG1_COL in optional_features:
        frame[PRESCREEN_LAG1_COL] = frame["Prescreen"].shift(1).fillna(0.0)
    if DSP_LAG1_COL in optional_features:
        frame[DSP_LAG1_COL] = frame["DSP"].shift(1).fillna(0.0)
    if PAID_SEARCH_LAG1_COL in optional_features:
        frame[PAID_SEARCH_LAG1_COL] = frame["Paid Search"].shift(1).fillna(0.0)
    if DSP_TRAILING_4W_AVG_COL in optional_features:
        frame[DSP_TRAILING_4W_AVG_COL] = frame["DSP"].rolling(window=4, min_periods=1).mean()
    if PAID_SEARCH_TRAILING_4W_AVG_COL in optional_features:
        frame[PAID_SEARCH_TRAILING_4W_AVG_COL] = frame["Paid Search"].rolling(window=4, min_periods=1).mean()
    if PRESCREEN_TRAILING_4W_AVG_COL in optional_features:
        frame[PRESCREEN_TRAILING_4W_AVG_COL] = frame["Prescreen"].rolling(window=4, min_periods=1).mean()
    if TARGET_LAG1_COL in optional_features:
        frame[TARGET_LAG1_COL] = lagged_target.fillna(0.0)
    if TARGET_TRAILING_4W_AVG_COL in optional_features:
        frame[TARGET_TRAILING_4W_AVG_COL] = lagged_target.rolling(window=4, min_periods=1).mean().fillna(0.0)

    return frame


def build_design_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_name: str,
    media_predictors: Sequence[str],
    media_transform_config: Dict[str, Dict[str, Any]],
    optional_features: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, object, List[str], Optional[str], Dict[str, str]]:
    """Build aligned train and test design matrices for one model run.

    Default media predictors:
        - DSP, LeadGen, Paid Search, Paid Social, Prescreen, Referrals

    Users can provide any subset of those six media variables via ``media_predictors``.
    Selected media variables can also be transformed through ``media_transform_config``.
    A transformed media variable replaces its raw version in the design matrix.

    Seasonal feature runs:
        - ``weekly``: uses ``W_*`` dummies with one dummy dropped
        - ``f_dummy``: uses ``F_*`` dummies with one dummy dropped
        - ``fourier``: uses ``sin_1``, ``cos_1``, ``sin_2``, ``cos_2`` and excludes dummy families

    Optional engineered variables that can be added on top of the defaults:
        - ``time_index``
        - ``time_index_sq``
        - ``year_indicator_2025``
        - ``year_indicator_2026``
        - ``Prescreen_lag1``
        - ``DSP_lag1``
        - ``Paid_Search_lag1``
        - ``DSP_trailing_4w_avg``
        - ``Paid_Search_trailing_4w_avg``
        - ``Prescreen_trailing_4w_avg``
        - ``APPLICATIONS_lag1`` or ``NON_DM_APPLICATIONS_lag1`` depending on the active target
        - ``APPLICATIONS_trailing_4w_avg`` or ``NON_DM_APPLICATIONS_trailing_4w_avg`` depending on the active target
    """
    train_df, test_df, media_feature_cols, media_feature_map = transform_media_variables(
        train_df=train_df,
        test_df=test_df,
        media_predictors=media_predictors,
        media_transform_config=media_transform_config,
    )
    run_cols, dropped_col = select_run_columns(pd.concat([train_df, test_df], axis=0), run_name)
    feature_cols = list(media_feature_cols) + list(optional_features) + run_cols
    feature_cols = [col for col in feature_cols if train_df[col].nunique(dropna=False) > 1]

    x_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].astype(float).copy()
    x_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].astype(float).copy()

    scaler_type = FEATURE_RUNS[run_name]["scaler"]
    scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
    scale_cols = list(media_feature_cols) + list(optional_features)
    if run_name == "fourier":
        scale_cols.extend(run_cols)
    scale_cols = list(dict.fromkeys(scale_cols))
    scale_cols = [col for col in scale_cols if col in x_train.columns]

    if scale_cols:
        x_train.loc[:, scale_cols] = scaler.fit_transform(x_train[scale_cols])
        x_test.loc[:, scale_cols] = scaler.transform(x_test[scale_cols])

    return x_train, y_train, x_test, y_test, scaler, feature_cols, dropped_col, media_feature_map


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = math.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / np.clip(np.abs(y_true_arr), EPSILON, None))) * 100.0)

    metrics = {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
    }

    if len(y_true_arr) >= 2 and not np.allclose(y_true_arr, y_true_arr[0]):
        metrics["R2_test"] = r2_score(y_true_arr, y_pred_arr)
    else:
        metrics["R2_test"] = np.nan

    return metrics


def adjusted_r2(r2_value: float, n_obs: int, n_predictors: int) -> float:
    if n_obs <= n_predictors + 1:
        return np.nan
    return 1.0 - (1.0 - r2_value) * ((n_obs - 1) / (n_obs - n_predictors - 1))


def compute_information_criteria(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_params: int,
) -> Tuple[float, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n_obs = len(y_true_arr)

    if n_obs == 0:
        return np.nan, np.nan

    rss = float(np.sum(np.square(y_true_arr - y_pred_arr)))
    rss = max(rss, EPSILON)
    aic = n_obs * math.log(rss / n_obs) + 2 * n_params
    bic = n_obs * math.log(rss / n_obs) + math.log(n_obs) * n_params
    return aic, bic


def fit_ols(x_train: pd.DataFrame, y_train: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    x_train_with_const = sm.add_constant(x_train, has_constant="add")
    return sm.OLS(y_train, x_train_with_const, missing="drop").fit()


def fit_nnls(x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, object]:
    x_values = x_train.to_numpy(dtype=float)
    y_values = y_train.to_numpy(dtype=float)
    coef, residual_norm = nnls(x_values, y_values)

    train_pred = x_values @ coef
    train_r2 = r2_score(y_values, train_pred) if len(y_values) >= 2 else np.nan
    n_obs = len(y_values)
    n_predictors = x_values.shape[1]
    aic, bic = compute_information_criteria(y_values, train_pred, n_predictors)

    return {
        "coef": coef,
        "intercept": 0.0,
        "residual_norm": float(residual_norm),
        "train_pred": train_pred,
        "train_r2": train_r2,
        "train_adj_r2": adjusted_r2(train_r2, n_obs, n_predictors),
        "aic": aic,
        "bic": bic,
        "feature_names": list(x_train.columns),
        "n_obs": n_obs,
        "n_predictors": n_predictors,
    }


def predict_nnls(nnls_result: Dict[str, object], x_frame: pd.DataFrame) -> np.ndarray:
    feature_names = nnls_result["feature_names"]
    coef = np.asarray(nnls_result["coef"], dtype=float)
    x_aligned = x_frame.reindex(columns=feature_names, fill_value=0.0)
    return x_aligned.to_numpy(dtype=float) @ coef


def model_diagnostics_row(
    metadata: RunMetadata,
    train_y: pd.Series,
    train_pred: Sequence[float],
    test_y: pd.Series,
    test_pred: Sequence[float],
    train_r2: float,
    train_adj_r2: float,
    aic: float,
    bic: float,
    spend_coefficients: str,
) -> Dict[str, object]:
    test_metrics = regression_metrics(test_y, test_pred)
    average_bias = float(np.mean(np.asarray(test_pred, dtype=float) - np.asarray(test_y, dtype=float)))
    row = asdict(metadata)
    row.update(
        {
            "n_observations": metadata.train_rows,
            "n_test_observations": metadata.test_rows,
            "R2": train_r2,
            "AdjR2": train_adj_r2,
            "MAE": test_metrics["MAE"],
            "MAPE": test_metrics["MAPE"],
            "RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2_test"],
            "Average_Bias": average_bias,
            "AIC": aic,
            "BIC": bic,
            "Spend_Coefficients": spend_coefficients,
        }
    )
    return row


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def entity_artifact_dir(output_root: Path, metadata: RunMetadata) -> Path:
    """Original per-entity artifact layout."""
    return output_root / metadata.dummy_family / metadata.scope / safe_name(metadata.entity) / metadata.model_type


def grouped_artifact_path(output_root: Path, metadata: RunMetadata, artifact_name: str) -> Path:
    """Grouped artifact layout for cross-entity comparison by artifact type.

    Files are stored directly in the artifact folder, with the entity name embedded in the
    filename so users do not need to click through one subfolder per state or division.
    """
    return (
        output_root
        / "by_artifact"
        / metadata.dummy_family
        / metadata.scope
        / metadata.model_type
        / artifact_name
    )


def scaler_metadata(scaler: object, scaler_type: str) -> Dict[str, object]:
    if not hasattr(scaler, "feature_names_in_"):
        return {
            "scaler_type": scaler_type,
            "scaled_columns": [],
        }

    feature_names = scaler.feature_names_in_.tolist()
    if scaler_type == "minmax":
        return {
            "scaler_type": scaler_type,
            "scaled_columns": feature_names,
            "scaler_data_min": dict(zip(feature_names, scaler.data_min_.tolist())),
            "scaler_data_max": dict(zip(feature_names, scaler.data_max_.tolist())),
        }

    return {
        "scaler_type": scaler_type,
        "scaled_columns": feature_names,
        "scaler_mean": dict(zip(feature_names, scaler.mean_.tolist())),
        "scaler_scale": dict(zip(feature_names, scaler.scale_.tolist())),
    }


def format_spend_coefficients(
    coefficient_map: Dict[str, float],
    media_feature_map: Dict[str, str],
) -> str:
    """Format the modeled spend coefficients for inclusion in consolidated diagnostics."""
    parts = []
    for raw_name, modeled_name in media_feature_map.items():
        if modeled_name in coefficient_map:
            parts.append(f"{raw_name}->{modeled_name}={float(coefficient_map[modeled_name]):.6g}")
    return "; ".join(parts)


def save_ols_artifacts(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    scaler: object,
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_model_path = grouped_artifact_path(output_root, metadata, "model") / f"{entity_prefix}_model.pkl"
    grouped_scaler_path = grouped_artifact_path(output_root, metadata, "scaler") / f"{entity_prefix}_scaler.joblib"
    grouped_coef_path = grouped_artifact_path(output_root, metadata, "coefficients_with_pvalues") / f"{entity_prefix}_coefficients_with_pvalues.csv"
    grouped_summary_path = grouped_artifact_path(output_root, metadata, "statsmodels_summary") / f"{entity_prefix}_statsmodels_summary.txt"
    grouped_metadata_path = grouped_artifact_path(output_root, metadata, "run_metadata") / f"{entity_prefix}_run_metadata.json"
    grouped_model_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_coef_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_summary_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    result.save(str(artifact_dir / "model.pkl"))
    result.save(str(grouped_model_path))
    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(scaler, grouped_scaler_path)

    coefficients = pd.DataFrame(
        {
            "term": result.params.index,
            "coefficient": result.params.values,
            "p_value": result.pvalues.reindex(result.params.index).values,
            "std_error": result.bse.reindex(result.params.index).values,
            "t_value": result.tvalues.reindex(result.params.index).values,
        }
    )
    coefficients.to_csv(artifact_dir / "coefficients_with_pvalues.csv", index=False)
    coefficients.to_csv(grouped_coef_path, index=False)

    save_text(artifact_dir / "statsmodels_summary.txt", result.summary().as_text())
    save_text(grouped_summary_path, result.summary().as_text())
    metadata_payload = {
        **asdict(metadata),
        **scaler_metadata(scaler, metadata.scaler_type),
    }
    save_json(
        artifact_dir / "run_metadata.json",
        metadata_payload,
    )
    save_json(grouped_metadata_path, metadata_payload)


def save_nnls_artifacts(
    nnls_result: Dict[str, object],
    scaler: object,
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_model_path = grouped_artifact_path(output_root, metadata, "model") / f"{entity_prefix}_model.joblib"
    grouped_scaler_path = grouped_artifact_path(output_root, metadata, "scaler") / f"{entity_prefix}_scaler.joblib"
    grouped_coef_path = grouped_artifact_path(output_root, metadata, "coefficients") / f"{entity_prefix}_coefficients.csv"
    grouped_metadata_path = grouped_artifact_path(output_root, metadata, "run_metadata") / f"{entity_prefix}_run_metadata.json"
    grouped_model_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_scaler_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_coef_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, artifact_dir / "scaler.joblib")
    joblib.dump(scaler, grouped_scaler_path)
    joblib.dump(nnls_result, artifact_dir / "model.joblib")
    joblib.dump(nnls_result, grouped_model_path)

    coefficients = pd.DataFrame(
        {
            "term": nnls_result["feature_names"],
            "coefficient": np.asarray(nnls_result["coef"], dtype=float),
        }
    )
    coefficients.to_csv(artifact_dir / "coefficients.csv", index=False)
    coefficients.to_csv(grouped_coef_path, index=False)

    metadata_payload = {
        **asdict(metadata),
        "intercept": nnls_result["intercept"],
        "residual_norm": nnls_result["residual_norm"],
        **scaler_metadata(scaler, metadata.scaler_type),
    }
    save_json(artifact_dir / "run_metadata.json", metadata_payload)
    save_json(grouped_metadata_path, metadata_payload)


def save_predictions(
    train_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    train_pred: Sequence[float],
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_predictions_path = grouped_artifact_path(output_root, metadata, "predictions") / f"{entity_prefix}_predictions.csv"
    grouped_predictions_path.parent.mkdir(parents=True, exist_ok=True)

    train_output = train_df[[YEAR_COL, WEEK_COL, TARGET_COL]].copy()
    train_output["prediction"] = np.asarray(train_pred, dtype=float)
    train_output["dataset"] = "train"

    prediction_frame = pd.concat([train_output, test_predictions_df], axis=0, ignore_index=True)
    prediction_frame.to_csv(artifact_dir / "predictions.csv", index=False)
    prediction_frame.to_csv(grouped_predictions_path, index=False)


def create_actual_vs_predicted_figure(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: Sequence[float],
    test_pred: Sequence[float],
    metadata: RunMetadata,
) -> Tuple[plt.Figure, plt.Axes]:
    train_weeks = train_df[YEAR_COL].astype(str) + "-W" + train_df[WEEK_COL].astype(str).str.zfill(2)
    test_weeks = test_df[YEAR_COL].astype(str) + "-W" + test_df[WEEK_COL].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(train_weeks, train_df[TARGET_COL], label="Train Actual", color="#1f77b4", linewidth=2)
    ax.plot(train_weeks, np.asarray(train_pred, dtype=float), label="Train Predicted", color="#ff7f0e", linestyle="--")
    ax.plot(test_weeks, test_df[TARGET_COL], label="Test Actual", color="#2ca02c", linewidth=2)
    ax.plot(test_weeks, np.asarray(test_pred, dtype=float), label="Test Predicted", color="#d62728", linestyle="--")
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(TARGET_COL)
    ax.set_title(f"Actual vs Predicted: {metadata.scope}={metadata.entity} | {metadata.model_type} | {metadata.dummy_family}")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_actual_vs_predicted(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: Sequence[float],
    test_pred: Sequence[float],
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_plot_path = grouped_artifact_path(output_root, metadata, "actual_vs_predicted") / f"{entity_prefix}_actual_vs_predicted.png"
    grouped_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, _ = create_actual_vs_predicted_figure(train_df, test_df, train_pred, test_pred, metadata)
    fig.savefig(artifact_dir / "actual_vs_predicted.png", dpi=150)
    fig.savefig(grouped_plot_path, dpi=150)
    plt.close(fig)


def plot_residuals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: Sequence[float],
    test_pred: Sequence[float],
    metadata: RunMetadata,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_plot_path = grouped_artifact_path(output_root, metadata, "residuals") / f"{entity_prefix}_residuals.png"
    grouped_plot_path.parent.mkdir(parents=True, exist_ok=True)

    train_residuals = train_df[TARGET_COL].to_numpy(dtype=float) - np.asarray(train_pred, dtype=float)
    test_residuals = test_df[TARGET_COL].to_numpy(dtype=float) - np.asarray(test_pred, dtype=float)

    plt.figure(figsize=(12, 6))
    plt.scatter(np.asarray(train_pred, dtype=float), train_residuals, label="Train", color="#1f77b4", alpha=0.75)
    plt.scatter(np.asarray(test_pred, dtype=float), test_residuals, label="Test", color="#d62728", alpha=0.85)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot: {metadata.scope}={metadata.entity} | {metadata.model_type} | {metadata.dummy_family}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_dir / "residuals.png", dpi=150)
    plt.savefig(grouped_plot_path, dpi=150)
    plt.close()


def build_contribution_frame(
    x_frame: pd.DataFrame,
    coefficients: Dict[str, float],
    intercept: float,
) -> pd.DataFrame:
    contribution_frame = x_frame.copy()

    contribution_cols: List[str] = []
    for col in x_frame.columns:
        coef_value = float(coefficients.get(col, 0.0))
        contribution_col = f"{col}__contribution"
        contribution_frame[contribution_col] = x_frame[col].astype(float) * coef_value
        contribution_cols.append(contribution_col)

    contribution_frame["Intercept__contribution"] = intercept
    contribution_cols.append("Intercept__contribution")
    contribution_frame["Predicted_Total"] = contribution_frame[contribution_cols].sum(axis=1)
    return contribution_frame


def save_contribution_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    metadata: RunMetadata,
    coefficients: Dict[str, float],
    intercept: float,
    output_root: Path,
) -> None:
    artifact_dir = entity_artifact_dir(output_root, metadata)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    entity_prefix = safe_name(metadata.entity)
    grouped_csv_path = grouped_artifact_path(output_root, metadata, "contribution_decomposition_csv") / f"{entity_prefix}_contribution_decomposition.csv"
    grouped_png_path = grouped_artifact_path(output_root, metadata, "contribution_decomposition") / f"{entity_prefix}_contribution_decomposition.png"
    grouped_csv_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_png_path.parent.mkdir(parents=True, exist_ok=True)

    train_contrib = build_contribution_frame(x_train, coefficients, intercept)
    train_contrib[YEAR_COL] = train_df[YEAR_COL].to_numpy()
    train_contrib[WEEK_COL] = train_df[WEEK_COL].to_numpy()
    train_contrib["dataset"] = "train"

    test_contrib = build_contribution_frame(x_test, coefficients, intercept)
    test_contrib[YEAR_COL] = test_df[YEAR_COL].to_numpy()
    test_contrib[WEEK_COL] = test_df[WEEK_COL].to_numpy()
    test_contrib["dataset"] = "test"

    contribution_df = pd.concat([train_contrib, test_contrib], axis=0, ignore_index=True)
    contribution_df.to_csv(artifact_dir / "contribution_decomposition.csv", index=False)
    contribution_df.to_csv(grouped_csv_path, index=False)

    contribution_columns = [
        col
        for col in contribution_df.columns
        if col.endswith("__contribution") and not col.startswith("Intercept__")
    ]

    contribution_summary = contribution_df.groupby("dataset")[contribution_columns + ["Intercept__contribution"]].sum().T
    contribution_summary.columns = [str(col).title() for col in contribution_summary.columns]
    contribution_summary["abs_total"] = contribution_summary.abs().sum(axis=1)
    contribution_summary = contribution_summary.sort_values("abs_total", ascending=False).drop(columns=["abs_total"])
    contribution_summary = contribution_summary.head(12)

    fig, ax = plt.subplots(figsize=(12, 7))
    contribution_summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("Contribution")
    ax.set_title(
        f"Contribution Decomposition: {metadata.scope}={metadata.entity} | {metadata.model_type} | {metadata.dummy_family}"
    )
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(artifact_dir / "contribution_decomposition.png", dpi=150)
    fig.savefig(grouped_png_path, dpi=150)
    plt.close(fig)


def display_inline_review(
    metadata: RunMetadata,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: Sequence[float],
    test_pred: Sequence[float],
    test_y: pd.Series,
    ols_result: Optional[sm.regression.linear_model.RegressionResultsWrapper] = None,
) -> None:
    """Display notebook-friendly inline output for a user-selected subset.

    Inline review is intended for interactive inspection in Jupyter. For OLS, the full
    statsmodels summary is shown.
    For all inline reviews, an actual-vs-predicted chart and an average-bias diagnostic
    are displayed.
    """
    average_bias = float(np.mean(np.asarray(test_pred, dtype=float) - np.asarray(test_y, dtype=float)))

    try:
        from IPython.display import Markdown, display

        display(
            Markdown(
                f"### {metadata.scope.title()}: `{metadata.entity}` | {metadata.model_type} | {metadata.dummy_family}\n"
                f"- Train rows: {metadata.train_rows}\n"
                f"- Test rows: {metadata.test_rows}\n"
                f"- Scaler: `{metadata.scaler_type}`\n"
                f"- Average Bias (`predicted - actual`): `{average_bias:.4f}`"
            )
        )
        if ols_result is not None:
            display(Markdown("#### Statsmodels Summary"))
            print(ols_result.summary().as_text())

        fig, _ = create_actual_vs_predicted_figure(train_df, test_df, train_pred, test_pred, metadata)
        display(fig)
        plt.close(fig)
    except ImportError:
        print(
            f"{metadata.scope.title()}: {metadata.entity} | {metadata.model_type} | {metadata.dummy_family} | "
            f"Average Bias (predicted - actual): {average_bias:.4f}"
        )
        if ols_result is not None:
            print(ols_result.summary().as_text())


def run_scope(
    df: pd.DataFrame,
    scope_col: str,
    scope_name: str,
    output_root: Optional[Path],
    model_types: Set[str],
    feature_runs: Set[str],
    media_predictors: Sequence[str],
    media_transform_config: Dict[str, Dict[str, Any]],
    selected_entities: Optional[Sequence[str]],
    optional_features: Sequence[str],
    inline_output: bool,
    backtest_mode: str,
    fixed_window_weeks: int,
) -> List[Dict[str, object]]:
    """Run the requested model families for one scope.

    Parameters:
        df: Full validated modeling dataset.
        scope_col: ``STATE_CD`` or ``Division``.
        scope_name: Human-readable scope label used in outputs.
        output_root: Root folder for saved artifacts. Pass ``None`` when using inline notebook mode.
        model_types: Subset of ``{"OLS", "NNLS"}``.
        feature_runs: Subset of ``{"weekly", "f_dummy", "fourier"}``.
        media_predictors: Selected subset of the six base media variables.
        media_transform_config: Optional per-media transform settings. Any transformed media
            variable replaces its raw form in the design matrix.
        selected_entities: Optional list of states or divisions to run.
        optional_features: Optional engineered variables to include in addition to the default predictors.
        inline_output: If ``True``, show notebook-friendly summaries instead of writing files.
        backtest_mode: Backtest style used for evaluation.
        fixed_window_weeks: Training window size for fixed-window rolling backtests.
    """
    diagnostics: List[Dict[str, object]] = []
    scope_df = prepare_entity_subset(df, scope_col, selected_entities)

    for run_name, run_config in FEATURE_RUNS.items():
        if run_name not in feature_runs:
            continue

        for entity, entity_df in scope_df.groupby(scope_col, dropna=False):
            modeling_df = entity_df
            if scope_name == "division":
                modeling_df = aggregate_division_weekly(entity_df, run_name)
            modeling_df = add_optional_features(modeling_df, optional_features)

            backtest_splits = generate_backtest_splits(modeling_df, backtest_mode, fixed_window_weeks)
            if not backtest_splits:
                continue

            entity_name = str(entity)

            if "OLS" in model_types:
                ols_result = None
                ols_last_scaler = None
                ols_last_train_df = None
                ols_last_x_train = None
                ols_last_test_df = None
                ols_last_x_test = None
                ols_last_train_pred = None
                ols_test_frames: List[pd.DataFrame] = []

                for train_df, test_df in backtest_splits:
                    x_train, y_train, x_test, y_test, scaler, predictors, dropped_dummy, media_feature_map = build_design_matrices(
                        train_df=train_df,
                        test_df=test_df,
                        run_name=run_name,
                        media_predictors=media_predictors,
                        media_transform_config=media_transform_config,
                        optional_features=optional_features,
                    )
                    if x_train.empty or len(y_train) <= x_train.shape[1] + 1:
                        continue

                    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
                    current_result = fit_ols(x_train, y_train)
                    current_train_pred = current_result.predict(sm.add_constant(x_train, has_constant="add"))
                    current_test_pred = current_result.predict(
                        sm.add_constant(x_test, has_constant="add").reindex(current_result.model.exog_names, axis=1, fill_value=0.0)
                    )

                    test_output = test_df[[YEAR_COL, WEEK_COL, TARGET_COL]].copy()
                    test_output["prediction"] = np.asarray(current_test_pred, dtype=float)
                    test_output["dataset"] = "test"
                    ols_test_frames.append(test_output)

                    ols_result = current_result
                    ols_last_scaler = scaler
                    ols_last_train_df = train_df.copy()
                    ols_last_x_train = x_train.copy()
                    ols_last_test_df = test_df.copy()
                    ols_last_x_test = x_test.copy()
                    ols_last_train_pred = np.asarray(current_train_pred, dtype=float)
                    ols_media_feature_map = dict(media_feature_map)

                if (
                    ols_result is None
                    or ols_last_scaler is None
                    or ols_last_train_df is None
                    or ols_last_x_train is None
                    or ols_last_test_df is None
                    or ols_last_x_test is None
                ):
                    continue

                ols_test_output = pd.concat(ols_test_frames, axis=0, ignore_index=True).sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)
                y_test = ols_test_output[TARGET_COL].astype(float)
                ols_test_pred = ols_test_output["prediction"].astype(float).to_numpy()
                ols_metadata = RunMetadata(
                    scope=scope_name,
                    entity=entity_name,
                    model_type="OLS",
                    dummy_family=run_name,
                    dropped_dummy=dropped_dummy,
                    train_rows=len(ols_last_train_df),
                    test_rows=len(ols_test_output),
                    predictors=predictors,
                    scaler_type=str(run_config["scaler"]),
                    backtest_mode=backtest_mode,
                    media_transform_config=media_transform_config,
                )
                diagnostics.append(
                    model_diagnostics_row(
                        metadata=ols_metadata,
                        train_y=ols_last_train_df[TARGET_COL].astype(float),
                        train_pred=ols_last_train_pred,
                        test_y=y_test,
                        test_pred=ols_test_pred,
                        train_r2=float(ols_result.rsquared),
                        train_adj_r2=float(ols_result.rsquared_adj),
                        aic=float(ols_result.aic),
                        bic=float(ols_result.bic),
                        spend_coefficients=format_spend_coefficients(
                            {name: value for name, value in ols_result.params.items() if name != "const"},
                            ols_media_feature_map,
                        ),
                    )
                )
                if inline_output:
                    display_inline_review(
                        ols_metadata,
                        ols_last_train_df,
                        ols_test_output,
                        ols_last_train_pred,
                        ols_test_pred,
                        y_test,
                        ols_result,
                    )
                else:
                    save_ols_artifacts(ols_result, ols_last_scaler, ols_metadata, output_root)
                    save_predictions(ols_last_train_df, ols_test_output, ols_last_train_pred, ols_metadata, output_root)
                    plot_actual_vs_predicted(ols_last_train_df, ols_test_output, ols_last_train_pred, ols_test_pred, ols_metadata, output_root)
                    plot_residuals(ols_last_train_df, ols_test_output, ols_last_train_pred, ols_test_pred, ols_metadata, output_root)
                    save_contribution_outputs(
                        train_df=ols_last_train_df,
                        test_df=ols_last_test_df,
                        x_train=ols_last_x_train,
                        x_test=ols_last_x_test.reindex(columns=ols_last_x_train.columns, fill_value=0.0),
                        metadata=ols_metadata,
                        coefficients={name: value for name, value in ols_result.params.items() if name != "const"},
                        intercept=float(ols_result.params.get("const", 0.0)),
                        output_root=output_root,
                    )

            if "NNLS" in model_types:
                nnls_result = None
                nnls_last_scaler = None
                nnls_last_train_df = None
                nnls_last_x_train = None
                nnls_last_test_df = None
                nnls_last_x_test = None
                nnls_last_train_pred = None
                nnls_test_frames: List[pd.DataFrame] = []

                for train_df, test_df in backtest_splits:
                    x_train, y_train, x_test, y_test, scaler, predictors, dropped_dummy, media_feature_map = build_design_matrices(
                        train_df=train_df,
                        test_df=test_df,
                        run_name=run_name,
                        media_predictors=media_predictors,
                        media_transform_config=media_transform_config,
                        optional_features=optional_features,
                    )
                    if x_train.empty:
                        continue
                    x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
                    current_result = fit_nnls(x_train, y_train)
                    current_train_pred = np.asarray(current_result["train_pred"], dtype=float)
                    current_test_pred = predict_nnls(current_result, x_test)

                    test_output = test_df[[YEAR_COL, WEEK_COL, TARGET_COL]].copy()
                    test_output["prediction"] = np.asarray(current_test_pred, dtype=float)
                    test_output["dataset"] = "test"
                    nnls_test_frames.append(test_output)

                    nnls_result = current_result
                    nnls_last_scaler = scaler
                    nnls_last_train_df = train_df.copy()
                    nnls_last_x_train = x_train.copy()
                    nnls_last_test_df = test_df.copy()
                    nnls_last_x_test = x_test.copy()
                    nnls_last_train_pred = current_train_pred
                    nnls_media_feature_map = dict(media_feature_map)

                if (
                    nnls_result is None
                    or nnls_last_scaler is None
                    or nnls_last_train_df is None
                    or nnls_last_x_train is None
                    or nnls_last_test_df is None
                    or nnls_last_x_test is None
                ):
                    continue

                nnls_test_output = pd.concat(nnls_test_frames, axis=0, ignore_index=True).sort_values([YEAR_COL, WEEK_COL]).reset_index(drop=True)
                y_test = nnls_test_output[TARGET_COL].astype(float)
                nnls_test_pred = nnls_test_output["prediction"].astype(float).to_numpy()
                nnls_metadata = RunMetadata(
                    scope=scope_name,
                    entity=entity_name,
                    model_type="NNLS",
                    dummy_family=run_name,
                    dropped_dummy=dropped_dummy,
                    train_rows=len(nnls_last_train_df),
                    test_rows=len(nnls_test_output),
                    predictors=predictors,
                    scaler_type=str(run_config["scaler"]),
                    backtest_mode=backtest_mode,
                    media_transform_config=media_transform_config,
                )
                diagnostics.append(
                    model_diagnostics_row(
                        metadata=nnls_metadata,
                        train_y=nnls_last_train_df[TARGET_COL].astype(float),
                        train_pred=nnls_last_train_pred,
                        test_y=y_test,
                        test_pred=nnls_test_pred,
                        train_r2=float(nnls_result["train_r2"]),
                        train_adj_r2=float(nnls_result["train_adj_r2"]),
                        aic=float(nnls_result["aic"]),
                        bic=float(nnls_result["bic"]),
                        spend_coefficients=format_spend_coefficients(
                            dict(zip(nnls_result["feature_names"], np.asarray(nnls_result["coef"], dtype=float))),
                            nnls_media_feature_map,
                        ),
                    )
                )
                if inline_output:
                    display_inline_review(
                        nnls_metadata,
                        nnls_last_train_df,
                        nnls_test_output,
                        nnls_last_train_pred,
                        nnls_test_pred,
                        y_test,
                    )
                else:
                    save_nnls_artifacts(nnls_result, nnls_last_scaler, nnls_metadata, output_root)
                    save_predictions(nnls_last_train_df, nnls_test_output, nnls_last_train_pred, nnls_metadata, output_root)
                    plot_actual_vs_predicted(nnls_last_train_df, nnls_test_output, nnls_last_train_pred, nnls_test_pred, nnls_metadata, output_root)
                    plot_residuals(nnls_last_train_df, nnls_test_output, nnls_last_train_pred, nnls_test_pred, nnls_metadata, output_root)
                    save_contribution_outputs(
                        train_df=nnls_last_train_df,
                        test_df=nnls_last_test_df,
                        x_train=nnls_last_x_train,
                        x_test=nnls_last_x_test.reindex(columns=nnls_last_x_train.columns, fill_value=0.0),
                        metadata=nnls_metadata,
                        coefficients=dict(zip(nnls_result["feature_names"], np.asarray(nnls_result["coef"], dtype=float))),
                        intercept=float(nnls_result["intercept"]),
                        output_root=output_root,
                    )

    return diagnostics


def run_model_pipeline(
    input_path: str = "/Users/Rahul/Desktop/Code/Working Codebase/ModelingFile_Digital.csv",
    output_dir: Optional[str] = "state_division_model_artifacts",
    selected_states: Optional[Sequence[str]] = None,
    selected_divisions: Optional[Sequence[str]] = None,
    methodologies: Optional[Sequence[str]] = None,
    media_predictors: Optional[Sequence[str]] = None,
    media_transform_config: Optional[Dict[str, Dict[str, Any]]] = None,
    optional_features: Optional[Sequence[str]] = None,
    inline_output: bool = False,
    backtest_mode: str = "fixed_holdout",
    fixed_window_weeks: int = DEFAULT_FIXED_WINDOW_WEEKS,
) -> pd.DataFrame:
    """
    Run the full modeling workflow and return the consolidated diagnostics DataFrame.

    Default behavior:
        Writes model artifacts to ``output_dir`` exactly as before, and also creates
        grouped comparison folders under ``output_dir/by_artifact`` so the same artifact
        type for all states or divisions sits together for easier review.

    Notebook inline review mode:
        Set ``inline_output=True`` to display detailed OLS statsmodels summaries, actual-vs-predicted
        charts, and average-bias diagnostics directly inside Jupyter instead of writing files.

    Parameters:
        input_path: CSV file to model.
        output_dir: Folder for saved artifacts. Leave as the default for file outputs. Pass ``None``
            or keep it unused when ``inline_output=True``.
        selected_states: Optional list of state codes to run. When omitted, all eligible states run.
        selected_divisions: Optional list of division names to run. When omitted, all eligible divisions run.
        methodologies: Optional list controlling model types and feature runs.
            Supported values:
                - ``OLS``
                - ``NNLS``
                - ``weekly``
                - ``f_dummy``
                - ``Fourier`` or ``fourier``
        media_predictors: Optional subset of the six base media variables. If omitted, the
            script uses all six by default:
                - ``DSP``
                - ``LeadGen``
                - ``Paid Search``
                - ``Paid Social``
                - ``Prescreen``
                - ``Referrals``
        media_transform_config: Optional per-media transform settings. Example:
            ``{"DSP": {"alpha": 0.5, "saturation": "log1p"}, "Prescreen": {"alpha": 0.7}}``
            Any media variable not listed here is used raw. Any media variable listed here is
            used only in transformed form.
        optional_features: Optional engineered variables to add on top of the default predictors.
            Available options:
                - ``time_index``
                - ``time_index_sq``
                - ``year_indicator_2025``
                - ``year_indicator_2026``
                - ``Prescreen_lag1``
                - ``DSP_lag1``
                - ``Paid_Search_lag1``
                - ``DSP_trailing_4w_avg``
                - ``Paid_Search_trailing_4w_avg``
                - ``Prescreen_trailing_4w_avg``
                - ``APPLICATIONS_lag1`` or ``NON_DM_APPLICATIONS_lag1`` depending on the active target
                - ``APPLICATIONS_trailing_4w_avg`` or ``NON_DM_APPLICATIONS_trailing_4w_avg`` depending on the active target
        inline_output: If ``True``, show inline notebook output and skip writing model folders.
        backtest_mode: One of ``fixed_holdout``, ``rolling_one_step_expanding``, or
            ``rolling_one_step_fixed_window``.
        fixed_window_weeks: Number of prior rows retained for the fixed-window rolling backtest.

    Example for Jupyter:
        from build_state_division_models import run_model_pipeline
        diagnostics_df = run_model_pipeline(
            selected_states=["CA", "TX"],
            methodologies=["OLS", "Fourier"],
            media_predictors=["DSP", "Prescreen", "Paid Search"],
            media_transform_config={"DSP": {"alpha": 0.5, "saturation": "log1p"}},
            optional_features=["time_index", "Prescreen_lag1"],
            inline_output=True,
            output_dir=None,
            backtest_mode="rolling_one_step_expanding",
        )
    """
    validated_media_predictors = validate_media_predictors(media_predictors)
    validated_media_transform_config = validate_media_transform_config(
        media_transform_config,
        validated_media_predictors,
    )
    validated_optional_features = validate_optional_features(optional_features)
    model_types, feature_runs = parse_methodology_selection(methodologies)
    if backtest_mode not in BACKTEST_MODES:
        raise ValueError(f"Unsupported backtest mode '{backtest_mode}'. Choices: {sorted(BACKTEST_MODES)}")

    if inline_output:
        selected_count = len(selected_states or []) + len(selected_divisions or [])
        if selected_count == 0:
            raise ValueError("Inline output requires at least one selected state or division.")
        output_root = None
    else:
        output_root = Path(output_dir or "state_division_model_artifacts")
        output_root.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)

    diagnostics_rows: List[Dict[str, object]] = []
    diagnostics_rows.extend(
        run_scope(
            df,
            STATE_COL,
            "state",
            output_root,
            model_types,
            feature_runs,
            validated_media_predictors,
            validated_media_transform_config,
            selected_states,
            validated_optional_features,
            inline_output,
            backtest_mode,
            fixed_window_weeks,
        )
    )
    diagnostics_rows.extend(
        run_scope(
            df,
            DIVISION_COL,
            "division",
            output_root,
            model_types,
            feature_runs,
            validated_media_predictors,
            validated_media_transform_config,
            selected_divisions,
            validated_optional_features,
            inline_output,
            backtest_mode,
            fixed_window_weeks,
        )
    )

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    if not diagnostics_df.empty:
        diagnostics_df = diagnostics_df.sort_values(
            ["dummy_family", "scope", "entity", "model_type"]
        ).reset_index(drop=True)
    if output_root is not None:
        diagnostics_df.to_csv(output_root / "consolidated_model_diagnostics.csv", index=False)

    if output_root is not None:
        with (output_root / "run_manifest.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "input_path": input_path,
                    "output_dir": str(output_root.resolve()),
                    "diagnostics_rows": len(diagnostics_df),
                    "feature_runs": sorted(feature_runs),
                    "model_types": sorted(model_types),
                    "media_predictors": validated_media_predictors,
                    "media_transform_config": validated_media_transform_config,
                    "optional_features": validated_optional_features,
                    "selected_states": list(selected_states or []),
                    "selected_divisions": list(selected_divisions or []),
                    "backtest_mode": backtest_mode,
                    "fixed_window_weeks": fixed_window_weeks,
                    "train_years": sorted(TRAIN_YEARS),
                    "test_year": TEST_YEAR,
                    "test_weeks": sorted(TEST_WEEKS),
                },
                handle,
            )

    return diagnostics_df


def main() -> None:
    args = parse_args()
    run_model_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        selected_states=args.selected_states,
        selected_divisions=args.selected_divisions,
        methodologies=args.methodologies,
        media_predictors=args.media_predictors,
        optional_features=args.optional_features,
        inline_output=args.inline_output,
        backtest_mode=args.backtest_mode,
        fixed_window_weeks=args.fixed_window_weeks,
    )


if __name__ == "__main__":
    main()
