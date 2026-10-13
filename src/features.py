"""Season-window weather aggregation, master-table joins, and engineered features.

``build_master_table`` is the single source of truth for the modeling table -- it
is called by notebook 01 (batch) and its engine is shared with the live demo
(``app/app.py``) through ``add_engineered_features`` and
``season_features_for_one_plot``, so the notebook and the app can never drift apart.

Growing-season rule
-------------------
A plot's growing-season window is the planting month plus the following three
months (4 months total), with year wraparound. Weather is joined at that
(region, year, month) grain and aggregated to one row per plot -- a
many-to-one join, so the plot row count never changes.

Derived stress thresholds
-------------------------
The exploratory analysis in notebook 02/03 found two clear non-linearities:
yield is flat across the cool-to-mild temperature range and then collapses
above roughly 21 degC, and it saturates once the season has received roughly
220 mm of rain. Two "stress" features make those thresholds explicit so the
model does not have to rediscover them from scratch.

Crop-optimum features
---------------------
Yield has a pronounced, *crop-specific* hump in altitude (maize peaks near
1,700 m, barley/wheat near 2,600 m) and a crop-specific response to season
temperature. Two per-crop optimum constants -- the yield-weighted mean altitude
and yield-weighted mean growing-season temperature of each crop -- are fitted on
the **training** file only (Rule 6) and then applied unchanged to the test file
and to the live demo. The deviations from those optima
(``altitude_dev_from_crop_optimum_m`` and its square, and
``temp_dev_from_crop_optimum_c``) are what the model actually consumes.
"""
import json
import os

import numpy as np
import pandas as pd

from cleaning import MONTH_ORDER, MONTH_NUM

SEASON_LENGTH = 4  # planting month + 3 following months

# Non-linear thresholds observed in the training data (see notebook 02 B3/B4).
GRAIN_HEAT_THRESHOLD_C = 21.0   # season mean temp above which yield collapses
SEASON_RAIN_SUFFICIENT_MM = 220.0  # season rainfall above which yield saturates

# Single source of truth for the model's feature contract -- notebook 04 (training)
# and app/app.py (the live demo) both import this, so they can never drift apart.
CATEGORICAL_FEATURES = ["region", "crop_type", "planting_month"]

# Features a user physically types in (or toggles) in the demo.
PLOT_NUMERIC_FEATURES = [
    "altitude_m", "rainfall_mm_season", "farm_size_ha", "fertilizer_kg_per_ha",
    "improved_seed_used", "pest_disease_flag", "soil_quality_index", "labor_days_per_ha",
    "distance_to_market_km",
]

# Features derived from the regional weather table's growing-season window.
WEATHER_FEATURES = [
    "season_avg_temp_c", "season_max_temp_c", "season_min_temp_c", "season_temp_std_c",
    "season_rainfall_mm_total", "season_rainfall_mm_wettest", "season_rainfall_mm_driest",
    "season_extreme_heat_days", "season_max_extreme_heat_days", "season_months_matched",
    "planting_month_temp_c", "planting_month_rainfall_mm",
    "preplant_temp_c", "preplant_rainfall_mm",
    "temp_deviation_from_region_avg",
    # weather x plot interactions and the weather side of the crop-optimum signal
    "altitude_x_season_temp", "altitude_dev_x_season_temp",
    "temp_dev_from_crop_optimum_c", "log_season_rainfall_mm_total",
]

# Explicit threshold / interaction / crop-optimum features built on top of the join.
ENGINEERED_FEATURES = [
    "temp_excess_above_threshold_c", "heat_stress_days_excess", "season_rain_deficit_mm",
    "fertilizer_x_improved_seed", "soil_x_fertilizer", "rainfall_x_temp",
    "planting_month_num", "survey_year",
    # crop-optimum altitude features (plot altitude vs that crop's best altitude)
    "altitude_dev_from_crop_optimum_m", "altitude_dev_from_crop_optimum_sq",
    # agronomic transformations and interactions
    "log_fertilizer_kg_per_ha", "fertilizer_x_labor", "soil_quality_squared",
    "seed_minus_pest",
]

NUMERIC_FEATURES = PLOT_NUMERIC_FEATURES + WEATHER_FEATURES + ENGINEERED_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "yield_tons_per_ha"

CROP_OPTIMA_PATH = os.path.join("app", "assets", "crop_optima.json")


def season_window_months(planting_month, year, n=SEASON_LENGTH, start_offset=0):
    """[(month_name, year), ...] for planting_month..+n-1, with year wraparound.

    start_offset=-1 gives the month immediately before planting (used for the
    pre-planting soil-moisture signal).
    """
    start = MONTH_NUM[planting_month] + start_offset
    window = []
    for offset in range(n):
        m = start + offset
        y = int(year)
        while m > 12:
            m -= 12
            y += 1
        while m < 1:
            m += 12
            y -= 1
        window.append((MONTH_ORDER[m - 1], y))
    return window


def build_weather_lookup(weather_clean):
    """(region, year, month) -> {avg_temp_c, monthly_rainfall_mm, extreme_heat_days}."""
    lookup = {}
    for r in weather_clean.itertuples(index=False):
        lookup[(r.region, int(r.year), r.month)] = {
            "avg_temp_c": r.avg_temp_c,
            "monthly_rainfall_mm": r.monthly_rainfall_mm,
            "extreme_heat_days": r.extreme_heat_days,
        }
    return lookup


def aggregate_season_window(lookup, region, year, planting_month):
    """Aggregate one plot's growing-season window from a prebuilt weather lookup."""
    def _collect(months):
        temps, rains, heats = [], [], []
        for month, yr in months:
            row = lookup.get((region, yr, month))
            if row is None:
                continue
            temps.append(row["avg_temp_c"])
            rains.append(row["monthly_rainfall_mm"])
            heats.append(row["extreme_heat_days"])
        return temps, rains, heats

    temps, rains, heats = _collect(season_window_months(planting_month, year))
    matched = len(temps)

    pm_temps, pm_rains, pm_heats = _collect(season_window_months(planting_month, year, n=1))
    pre_temps, pre_rains, _ = _collect(season_window_months(planting_month, year, n=1, start_offset=-1))

    def _mean(v):
        return float(np.mean(v)) if v else np.nan

    def _std(v):
        return float(np.std(v)) if v else np.nan

    return {
        # summary statistics over the full 4-month window
        "season_avg_temp_c": _mean(temps),
        "season_max_temp_c": float(np.max(temps)) if temps else np.nan,
        "season_min_temp_c": float(np.min(temps)) if temps else np.nan,
        "season_temp_std_c": _std(temps),
        "season_rainfall_mm_total": float(np.sum(rains)) if rains else np.nan,
        "season_rainfall_mm_wettest": float(np.max(rains)) if rains else np.nan,
        "season_rainfall_mm_driest": float(np.min(rains)) if rains else np.nan,
        "season_extreme_heat_days": float(np.sum(heats)) if heats else np.nan,
        "season_max_extreme_heat_days": float(np.max(heats)) if heats else np.nan,
        "season_months_matched": int(matched),
        # the planting month on its own
        "planting_month_temp_c": _mean(pm_temps),
        "planting_month_rainfall_mm": _mean(pm_rains),
        # the month before planting
        "preplant_temp_c": _mean(pre_temps),
        "preplant_rainfall_mm": _mean(pre_rains),
    }


def aggregate_season_weather(weather_clean, region, year, planting_month):
    """Convenience wrapper used by notebook 01's join audit / join proof cells."""
    return aggregate_season_window(build_weather_lookup(weather_clean), region, year, planting_month)


def build_region_temp_climatology(weather_clean):
    return weather_clean.groupby("region")["avg_temp_c"].mean().to_dict()


def build_region_rain_climatology(weather_clean):
    return (weather_clean.groupby("region")["monthly_rainfall_mm"].mean() * SEASON_LENGTH).to_dict()


def compute_crop_optima(joined_train):
    """Per-crop yield-optimal altitude (m) and season temperature (degC).

    Both are yield-weighted means computed on the **training** table only: each
    plot's altitude/temperature is weighted by its own yield, so the result is the
    altitude (and temperature) at which that crop is, on average, most productive.
    The five constants per optimum are then reused verbatim for the test file and
    the live demo, which is why they are fitted here rather than inside the model.
    """
    optima = {"altitude_m": {}, "season_avg_temp_c": {}}
    for crop, g in joined_train.groupby("crop_type"):
        weights = g[TARGET_COLUMN].to_numpy(dtype=float)
        if weights.sum() > 0:
            optima["altitude_m"][crop] = float(np.average(g["altitude_m"], weights=weights))
            optima["season_avg_temp_c"][crop] = float(
                np.average(g["season_avg_temp_c"], weights=weights))
        else:
            optima["altitude_m"][crop] = float(g["altitude_m"].mean())
            optima["season_avg_temp_c"][crop] = float(g["season_avg_temp_c"].mean())
    return optima


def save_crop_optima(optima, path=CROP_OPTIMA_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(optima, fh, indent=2)


def load_crop_optima(path=CROP_OPTIMA_PATH):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def add_engineered_features(df, region_clim, crop_optima):
    """Add every threshold, ratio, interaction and crop-optimum feature.

    Operates on a table that already carries the plot columns *and* the
    growing-season weather columns, so notebook 01 and the demo app produce
    byte-identical features for identical inputs.
    """
    opt_alt = df["crop_type"].map(crop_optima["altitude_m"])
    opt_temp = df["crop_type"].map(crop_optima["season_avg_temp_c"])

    df["temp_deviation_from_region_avg"] = df["season_avg_temp_c"] - df["region"].map(region_clim)
    df["temp_excess_above_threshold_c"] = (
        df["season_avg_temp_c"] - GRAIN_HEAT_THRESHOLD_C
    ).clip(lower=0)
    df["heat_stress_days_excess"] = (df["season_extreme_heat_days"] - 1).clip(lower=0)
    df["season_rain_deficit_mm"] = (
        SEASON_RAIN_SUFFICIENT_MM - df["season_rainfall_mm_total"]
    ).clip(lower=0)
    df["fertilizer_x_improved_seed"] = df["fertilizer_kg_per_ha"] * df["improved_seed_used"]
    df["soil_x_fertilizer"] = df["soil_quality_index"] * df["fertilizer_kg_per_ha"]
    df["rainfall_x_temp"] = df["season_rainfall_mm_total"] * df["season_avg_temp_c"]
    df["planting_month_num"] = df["planting_month"].map(MONTH_NUM)

    # crop-optimum altitude signal: how far the plot is from that crop's best altitude
    alt_dev = df["altitude_m"] - opt_alt
    df["altitude_dev_from_crop_optimum_m"] = alt_dev
    df["altitude_dev_from_crop_optimum_sq"] = -(alt_dev ** 2)

    # altitude x temperature, and the crop-specific temperature deviation
    df["altitude_x_season_temp"] = (df["altitude_m"] / 1000.0) * df["season_avg_temp_c"]
    df["altitude_dev_x_season_temp"] = alt_dev * df["season_avg_temp_c"]
    df["temp_dev_from_crop_optimum_c"] = df["season_avg_temp_c"] - opt_temp

    # agronomic transforms / interactions
    df["log_fertilizer_kg_per_ha"] = np.log1p(df["fertilizer_kg_per_ha"])
    df["log_season_rainfall_mm_total"] = np.log1p(df["season_rainfall_mm_total"])
    df["fertilizer_x_labor"] = df["fertilizer_kg_per_ha"] * df["labor_days_per_ha"]
    df["soil_quality_squared"] = df["soil_quality_index"] ** 2
    df["seed_minus_pest"] = df["improved_seed_used"] - df["pest_disease_flag"]
    return df


def build_master_table(plot_clean, weather_clean, crop_optima=None, return_optima=False):
    """Clean plot rows -> season join -> engineered features (one master table).

    When ``crop_optima`` is None (the training call) the per-crop optima are fitted
    from the table being built; pass the returned optima back in when building the
    test table so both files use identical constants (Rule 6).
    """
    region_clim = build_region_temp_climatology(weather_clean)
    region_rain_clim = build_region_rain_climatology(weather_clean)
    lookup = build_weather_lookup(weather_clean)

    season_records = [
        aggregate_season_window(lookup, row["region"], row["survey_year"], row["planting_month"])
        for _, row in plot_clean.iterrows()
    ]
    season_df = pd.DataFrame(season_records, index=plot_clean.index)

    df = pd.concat([plot_clean, season_df], axis=1)

    # The weather join leaves season features NaN only when a plot's window hit
    # no matching weather rows; fall back to that region's overall climatology
    # (documented in notebook 01's A4 join audit).
    df["season_avg_temp_c"] = df["season_avg_temp_c"].fillna(df["region"].map(region_clim))
    df["season_max_temp_c"] = df["season_max_temp_c"].fillna(df["season_avg_temp_c"])
    df["season_min_temp_c"] = df["season_min_temp_c"].fillna(df["season_avg_temp_c"])
    df["season_temp_std_c"] = df["season_temp_std_c"].fillna(0.0)
    df["season_rainfall_mm_total"] = df["season_rainfall_mm_total"].fillna(
        df["region"].map(region_rain_clim)
    )
    df["season_rainfall_mm_wettest"] = df["season_rainfall_mm_wettest"].fillna(
        df["season_rainfall_mm_total"] / SEASON_LENGTH
    )
    df["season_rainfall_mm_driest"] = df["season_rainfall_mm_driest"].fillna(
        df["season_rainfall_mm_total"] / SEASON_LENGTH
    )
    df["season_extreme_heat_days"] = df["season_extreme_heat_days"].fillna(0.0)
    df["season_max_extreme_heat_days"] = df["season_max_extreme_heat_days"].fillna(0.0)
    df["planting_month_temp_c"] = df["planting_month_temp_c"].fillna(df["season_avg_temp_c"])
    df["planting_month_rainfall_mm"] = df["planting_month_rainfall_mm"].fillna(
        df["season_rainfall_mm_total"] / SEASON_LENGTH
    )
    df["preplant_temp_c"] = df["preplant_temp_c"].fillna(df["season_avg_temp_c"])
    df["preplant_rainfall_mm"] = df["preplant_rainfall_mm"].fillna(
        df["season_rainfall_mm_total"] / SEASON_LENGTH
    )

    if crop_optima is None:
        crop_optima = compute_crop_optima(df)

    df = add_engineered_features(df, region_clim, crop_optima)
    df["temp_deviation_from_region_avg"] = df["temp_deviation_from_region_avg"].fillna(0.0)

    if return_optima:
        return df, crop_optima
    return df


def season_features_for_one_plot(weather_clean, region, year, planting_month):
    """Single-plot version of build_master_table's weather join + fallback, for the
    live demo. Mirrors the same climatology fallback rule (region's overall mean
    temp/rainfall when the season window has no matching weather rows)."""
    agg = aggregate_season_weather(weather_clean, region, year, planting_month)
    region_clim = build_region_temp_climatology(weather_clean)
    region_rain_clim = build_region_rain_climatology(weather_clean)
    fallback_temp = region_clim.get(region, 0.0)
    fallback_rain = region_rain_clim.get(region, 0.0)

    for key in ["season_avg_temp_c", "season_max_temp_c", "season_min_temp_c",
                "planting_month_temp_c", "preplant_temp_c"]:
        if np.isnan(agg[key]):
            agg[key] = fallback_temp
    if np.isnan(agg["season_temp_std_c"]):
        agg["season_temp_std_c"] = 0.0
    for key in ["season_rainfall_mm_wettest", "season_rainfall_mm_driest",
                "planting_month_rainfall_mm", "preplant_rainfall_mm"]:
        if np.isnan(agg[key]):
            agg[key] = fallback_rain / SEASON_LENGTH
    if np.isnan(agg["season_rainfall_mm_total"]):
        agg["season_rainfall_mm_total"] = fallback_rain
    for key in ["season_extreme_heat_days", "season_max_extreme_heat_days"]:
        if np.isnan(agg[key]):
            agg[key] = 0.0
    return agg


def attach_price(df, price_clean):
    """Join price onto a copy of df for revenue analysis/demo -- never feed into the model."""
    out = df.merge(
        price_clean[["region", "crop_type", "year", "price_birr_per_quintal"]],
        left_on=["region", "crop_type", "survey_year"],
        right_on=["region", "crop_type", "year"],
        how="left",
    ).drop(columns=["year"])
    if "yield_tons_per_ha" in out.columns:
        out["revenue_birr_per_ha"] = out["yield_tons_per_ha"] * 10 * out["price_birr_per_quintal"]
    return out
