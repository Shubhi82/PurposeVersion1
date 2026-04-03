import pandas as pd
import statsmodels.api as sm

from data_processing import get_modeling_dataset
from data_processing import load_originations_data


# =========================
# LOAD MARKETING DATA
# =========================
def load_marketing_data():
    df = pd.read_csv("Marketing_Spend_Data.csv")
    df.columns = df.columns.str.upper()
    return df


# =========================
# RUN OLS MODEL
# =========================
def run_model():
    marketing_df = load_marketing_data()

    df = get_modeling_dataset(marketing_df)

    # -------------------------
    # Define target
    # -------------------------
    y = df["TARGET"]

    # -------------------------
    # Define features
    # (exclude non-feature columns)
    # -------------------------
    exclude_cols = [
        "ISO_YEAR",
        "ISO_WEEK",
        "STATE_CD",
        "APPLICATIONS",
        "DM_APPLICATIONS",
        "NON_DM_APPLICATIONS",
        "TARGET"
    ]

    X = df.drop(columns=[col for col in exclude_cols if col in df.columns])

    # Fill missing values
    X = X.fillna(0)

    # Add intercept
    X = sm.add_constant(X)

    # -------------------------
    # Fit model
    # -------------------------
    model = sm.OLS(y, X).fit()

    print(model.summary())

    return model


if __name__ == "__main__":
    run_model()
