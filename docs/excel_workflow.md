# Excel Workflow for Marketing Analysis

This document explains the manual workflow behind the Streamlit app and the modeling dataset.

## 1. Filter by state and product

Start from the `All_Combos` sheet.

- Filter `STATE_CD` to the state you want to analyze.
- Filter `PRODUCT_CD` only if you want a product-specific cut.
- Leave `PRODUCT_CD` unfiltered when you want an all-product state view.

## 2. Split channels

Do not mix channels.

- Build one subset where `CHANNEL_CD = DIGITAL`
- Build one subset where `CHANNEL_CD = PHYSICAL`

Each subset keeps:

- `ISO_YEAR`
- `ISO_WEEK`
- `STATE_CD`
- `PRODUCT_CD`
- tactic spend columns:
  - `DSP`
  - `LeadGen`
  - `Paid Search`
  - `Paid Social`
  - `Prescreen`
  - `Referrals`
  - `Sweepstakes`
- `APPLICATIONS`

## 3. Aggregate weekly

For weekly analysis:

- Group by `ISO_YEAR`, `ISO_WEEK`, and `CHANNEL_CD`
- Sum all tactic spend columns
- Sum `APPLICATIONS`

This produces one weekly row per channel for the current filter.

## 4. Aggregate fortnightly

For fortnight analysis:

- Create a fortnight bucket as `((ISO_WEEK - 1) // 2) + 1`
- Group by `ISO_YEAR`, fortnight bucket, and `CHANNEL_CD`
- Sum all tactic spend columns
- Sum `APPLICATIONS`

This produces one fortnight row per channel for the current filter.

## 5. Prepare the modeling dataset

For each channel separately:

1. Keep the dependent variable as channel-specific applications.
2. Keep the independent variables as the same channel's tactic spend columns.
3. Sort the aggregated rows chronologically.
4. Create sequential time dummies named `F01`, `F02`, ..., `Fn`.
5. Drop `F01` so the model does not suffer from dummy-variable multicollinearity.

## 6. Fit the regression

Run one model for each channel:

- DIGITAL model:
  - Dependent variable: `Applications_DIGITAL`
  - Predictors: DIGITAL tactic spend + time dummies
- PHYSICAL model:
  - Dependent variable: `Applications_PHYSICAL`
  - Predictors: PHYSICAL tactic spend + time dummies

## 7. Evaluate

Use:

- `MAPE = mean(abs((Actual - Predicted) / Actual)) * 100`

Review:

- actual vs predicted applications over time
- coefficient signs and magnitudes by tactic
- channel-specific MAPE
