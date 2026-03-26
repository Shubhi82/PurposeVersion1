from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from utils import CHANNELS, TACTIC_COLUMNS, compute_mape, make_interpretation_sentence


@dataclass
class ChannelModelResult:
    channel: str
    fitted_frame: pd.DataFrame
    coefficients: pd.DataFrame
    mape: float
    narrative: list[str]
    features_used: list[str]
    periods_used: list[str]


def fit_channel_models(model_df: pd.DataFrame) -> dict[str, ChannelModelResult]:
    results: dict[str, ChannelModelResult] = {}
    for channel in CHANNELS:
        channel_df = model_df.loc[model_df["CHANNEL_CD"] == channel].copy()
        if channel_df.empty:
            continue
        results[channel] = fit_single_channel_model(channel_df, channel=channel)
    return results


def fit_single_channel_model(channel_df: pd.DataFrame, channel: str) -> ChannelModelResult:
    tactic_features = [
        column
        for column in TACTIC_COLUMNS
        if column in channel_df.columns and channel_df[column].fillna(0).abs().sum() > 0
    ]
    dummy_features = sorted(column for column in channel_df.columns if column.startswith("F"))
    features = tactic_features + dummy_features

    if not features:
        raise ValueError(f"No usable predictors were found for the {channel} model.")

    X = sm.add_constant(channel_df[features].astype(float), has_constant="add")
    y = channel_df["APPLICATIONS"].astype(float)
    model = sm.OLS(y, X).fit()

    predictions = model.predict(X).clip(lower=0)
    fitted_frame = channel_df[
        ["period_label", "period_start", "APPLICATIONS", "TOTAL_SPEND"]
    ].copy()
    fitted_frame["Predicted_Applications"] = predictions
    fitted_frame = fitted_frame.sort_values("period_start").reset_index(drop=True)

    coefficients = (
        pd.DataFrame(
            {
                "feature": model.params.index,
                "coefficient": model.params.values,
                "p_value": model.pvalues.values,
            }
        )
        .sort_values("feature")
        .reset_index(drop=True)
    )

    tactic_coefficients = coefficients.loc[coefficients["feature"].isin(tactic_features)].copy()
    tactic_coefficients["abs_coefficient"] = tactic_coefficients["coefficient"].abs()
    top_tactics = tactic_coefficients.sort_values("abs_coefficient", ascending=False).head(3)
    narrative = [
        make_interpretation_sentence(channel, row.feature, row.coefficient)
        for row in top_tactics.itertuples(index=False)
    ]

    return ChannelModelResult(
        channel=channel,
        fitted_frame=fitted_frame,
        coefficients=coefficients.drop(columns="abs_coefficient", errors="ignore"),
        mape=compute_mape(fitted_frame["APPLICATIONS"], fitted_frame["Predicted_Applications"]),
        narrative=narrative,
        features_used=features,
        periods_used=sorted(channel_df["time_bucket"].unique().tolist()),
    )
