# Pen Demand Forecasting

This is a demand forecasting pipeline I built for a client project — predicting weekly unit demand for pens across multiple SKUs and warehouses, 12 weeks into the future.

**Important: all data in this repo is synthetic.** It was generated to mimic the structure and general patterns of the real client dataset (same columns, same rough scale — 670 SKUs, 12 warehouses, 3 years of weekly history), but none of it is real. No client sales figures, no real brand names, no real pricing — none of it is in here. This was done on purpose so the project could be developed and shared openly on GitHub without touching anything confidential. Brand names in the data have also been anonymised to `brand_1`, `brand_2`, etc. as an extra layer of separation from anything real.

## What it does

Takes weekly sales history (date, SKU, warehouse, category, brand, colour, tip size, price, promotion flag, units sold) and:

1. Cleans and fills in any gaps in the weekly time series
2. Builds lag and rolling-average features, plus calendar features (this is where the trend/seasonality signal comes from)
3. Trains a single LightGBM model across every SKU/warehouse combination at once
4. Checks how accurate it is against held-out weeks and against naive baselines
5. Recursively forecasts 12 weeks forward for every SKU/warehouse pair
6. Exports the results as a plain CSV and a tidier Excel summary (total demand per SKU, and a SKU × warehouse breakdown)

## Project structure

Raw and processed data files are excluded from version control on purpose — they're large, and even though this particular file is synthetic, keeping the habit of never committing data files means real client data will never accidentally end up on GitHub either.

## Running it

```powershell
.venv\Scripts\Activate.ps1
python src\forecast_pipeline.py
```

Needs the packages in `requirements.txt` installed first (`pip install -r requirements.txt`), and expects the raw Excel file to be sitting in `data\raw\`.

## How good is it, honestly

The model beats a "just repeat last week" and "just repeat this week last year" baseline by a solid margin (roughly 49% WMAPE vs ~69% for the naive guesses), but this dataset has a lot of built-in randomness by design, so 49% is close to the practical ceiling here — not a sign the model is undertrained. On real client data with less synthetic noise, I'd expect this to do noticeably better without changing much of the approach.

## What's not in here yet

Price and promotions are currently just carried forward at their last known value for future weeks, since there's no real promo calendar to plug in. If a client can supply a planned pricing/promo schedule for the forecast period, that would be a genuinely useful upgrade rather than a cosmetic one.
