# Marketing Analytics and Channel-Specific Modeling

This repo now contains a Streamlit workflow built around `/Users/shubhivashistha/Downloads/All_States_modeling.xlsx` for state-level marketing analysis and regression modeling.

## Why DIGITAL and PHYSICAL are modeled separately

DIGITAL and PHYSICAL should not be mixed in one regression because they represent different acquisition systems, different spend patterns, and different response behaviors. A combined model would let DIGITAL spend explain PHYSICAL applications, or vice versa, which would blur the effect of each channel and make interpretation unreliable.

By fitting two separate models:

- `Applications_DIGITAL` is explained only by DIGITAL tactic spend.
- `Applications_PHYSICAL` is explained only by PHYSICAL tactic spend.
- Each model gets its own time fixed effects, so weekly or fortnightly seasonality is absorbed without contaminating the other channel.

## Regression interpretation

The app fits an OLS regression separately for each channel:

```text
Applications_channel = Intercept + tactic spend variables + time dummies
```

Time dummies are created as `F01`, `F02`, ..., `Fn`, with `F01` dropped to avoid multicollinearity.

Example interpretation:

```text
1 unit increase in DIGITAL DSP spend increases DIGITAL applications by X,
holding the other DIGITAL tactics and time dummies constant.
```

The coefficient table shown in the app lets you read each tactic in the same way. Positive coefficients indicate incremental lift in applications for that channel, while negative coefficients indicate a negative association after controlling for the other channel-specific tactics and time effects.

## Pipeline

The implementation is split into four modules:

- `streamlit_app.py`: Streamlit UI with three tabs, where Tab 3 contains the marketing analysis and modeling workflow.
- `data_processing.py`: workbook loading, state/product filtering, channel split, weekly or fortnight aggregation, and modeling-frame preparation.
- `modeling.py`: separate DIGITAL and PHYSICAL regressions, fitted-value generation, coefficient extraction, and MAPE calculation.
- `utils.py`: shared constants, time helpers, MAPE helper, and interpretation text.

## Streamlit app behavior

Tab 3 contains:

- Section 1: marketing spend consistency by tactic for the selected state, split into DIGITAL and PHYSICAL panels.
- Section 2: product-level spend consistency for the selected state and product, with DIGITAL vs PHYSICAL comparison.
- Section 3: applications consistency over time, split by channel.
- Modeling section: one regression for DIGITAL applications and one regression for PHYSICAL applications, each with actual vs predicted values and MAPE.

The aggregation switch supports both:

- `Weekly`
- `Fortnight`

## Excel workflow

The Excel preparation logic is documented in [docs/excel_workflow.md](docs/excel_workflow.md).

In short, the workflow is:

1. Filter the workbook by `STATE_CD` and optionally `PRODUCT_CD`.
2. Split rows into `DIGITAL` and `PHYSICAL`.
3. Aggregate tactic spend and applications at the selected time grain.
4. Build channel-specific modeling datasets with time dummies.
5. Fit one model per channel and evaluate using MAPE.

## How to run

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Launch the app:

```bash
streamlit run streamlit_app.py
```

By default the app looks for:

```text
/Users/shubhivashistha/Downloads/All_States_modeling.xlsx
```

You can also upload the workbook directly inside the app.
