"""Ethiopian Crop Yield & Revenue Predictor -- Deliverable E demo.

Run with:  streamlit run app/app.py

The user only ever selects region / crop / year / planting month and types plot
measurements. Growing-season weather features and the market price are looked up
automatically from the cleaned tables bundled in ``app/assets/`` -- no weather
number or price is ever typed in by hand.

Developed by Zedingil Zelalem.
"""
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cleaning import CANONICAL_CROPS, CANONICAL_REGIONS, MONTH_NUM, MONTH_ORDER
from features import (CATEGORICAL_FEATURES, ENGINEERED_FEATURES, MODEL_FEATURES,
                      PLOT_NUMERIC_FEATURES, WEATHER_FEATURES, add_engineered_features,
                      build_region_temp_climatology, compute_crop_optima, load_crop_optima,
                      season_features_for_one_plot)

AUTHOR = "Zedingil Zelalem"

APP_DIR = os.path.dirname(__file__)
ASSETS = os.path.join(APP_DIR, "assets")
MODELS = os.path.join(APP_DIR, "..", "models")
PROCESSED = os.path.join(APP_DIR, "..", "data", "processed")
REPORTS = os.path.join(APP_DIR, "..", "reports")

GREEN_DARK = "#14532d"
GREEN = "#2e7d32"
GREEN_LIGHT = "#66bb6a"
AMBER = "#f9a825"
SLATE = "#334155"

st.set_page_config(
    page_title="Ethiopian Crop Yield & Revenue Predictor | Zedingil Zelalem",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, #f4f8f4 0%, #ffffff 42%); }}
    .hero {{
        background: linear-gradient(120deg, {GREEN_DARK} 0%, {GREEN} 55%, {GREEN_LIGHT} 100%);
        border-radius: 18px; padding: 2.1rem 2.4rem; margin-bottom: 1.4rem;
        box-shadow: 0 12px 30px rgba(20, 83, 45, 0.28);
    }}
    .hero h1 {{ color: #ffffff; font-size: 2.15rem; margin: 0 0 .35rem 0; letter-spacing: -.4px; }}
    .hero p {{ color: #e8f5e9; font-size: 1.02rem; margin: 0; max-width: 60rem; }}
    .hero .badge {{
        display: inline-block; margin-top: .9rem; padding: .32rem .85rem;
        background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.42);
        border-radius: 999px; color: #fff; font-size: .82rem; letter-spacing: .3px;
    }}
    .kpi {{
        background: #ffffff; border: 1px solid #e2ebe2; border-left: 6px solid {GREEN};
        border-radius: 14px; padding: 1.05rem 1.2rem; height: 100%;
        box-shadow: 0 4px 14px rgba(15, 40, 20, .07);
    }}
    .kpi .label {{ color: #64748b; font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; }}
    .kpi .value {{ color: {GREEN_DARK}; font-size: 1.95rem; font-weight: 700; line-height: 1.15; }}
    .kpi .sub {{ color: #64748b; font-size: .8rem; }}
    .kpi.amber {{ border-left-color: {AMBER}; }}
    .kpi.amber .value {{ color: #92400e; }}
    .lookup {{
        background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px;
        padding: .85rem 1.05rem; color: {SLATE}; font-size: .88rem;
    }}
    .lookup b {{ color: {GREEN_DARK}; }}
    .footer {{
        margin-top: 2.4rem; padding-top: 1.1rem; border-top: 1px solid #e2e8f0;
        color: #64748b; font-size: .82rem; text-align: center;
    }}
    .footer b {{ color: {GREEN_DARK}; }}
    div[data-testid="stForm"] {{ border: 1px solid #e2ebe2; border-radius: 16px; padding: 1.2rem 1.2rem .4rem; background: #ffffff; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌾 Ethiopian Crop Yield &amp; Revenue Predictor</h1>
        <p>Estimate how much a smallholder plot will produce — and what that harvest is worth —
        before planting decisions are made. Growing-season weather and market prices are looked
        up for you; you never type a weather number or a price.</p>
        <span class="badge">Developed by <b>Zedingil Zelalem</b> · Qiyas AI Hackathon 2026</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_assets():
    model = joblib.load(os.path.join(MODELS, "final_model.joblib"))
    weather_clean = pd.read_csv(os.path.join(ASSETS, "weather_clean.csv"))
    price_clean = pd.read_csv(os.path.join(ASSETS, "price_clean.csv"))
    summary_path = os.path.join(ASSETS, "region_crop_summary.csv")
    if os.path.exists(summary_path):
        region_crop = pd.read_csv(summary_path)
    else:
        master = pd.read_csv(os.path.join(PROCESSED, "master_train.csv"))
        region_crop = (master.groupby(["region", "crop_type"])["yield_tons_per_ha"]
                       .agg(mean_yield="mean", median_yield="median", n_plots="count")
                       .reset_index())
    # crop optima are fitted on train in notebook 01 and bundled; recompute as a fallback
    optima_path = os.path.join(ASSETS, "crop_optima.json")
    if os.path.exists(optima_path):
        crop_optima = load_crop_optima(optima_path)
    else:
        master = pd.read_csv(os.path.join(PROCESSED, "master_train.csv"))
        crop_optima = compute_crop_optima(master)
    return model, weather_clean, price_clean, region_crop, crop_optima


@st.cache_data(show_spinner=False)
def load_model_card():
    path = os.path.join(REPORTS, "final_model_metrics.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


try:
    model, weather_clean, price_clean, region_crop, CROP_OPTIMA = load_assets()
except FileNotFoundError as exc:
    st.error(
        f"Required file missing: {exc}\n\nRun notebook 01 then notebook 04 first so that "
        "models/final_model.joblib and app/assets/weather_clean.csv + price_clean.csv exist.",
        icon=":material/error:",
    )
    st.stop()

card = load_model_card()
val = card.get("validation", {})
r2 = val.get("R2")
rmse = val.get("RMSE")
mae = val.get("MAE")
selected_model = card.get("selected_model", "gradient-boosted trees")

with st.sidebar:
    st.markdown("### 🌱 Model card")
    c1, c2 = st.columns(2)
    c1.metric("R² on held-out data", f"{r2:.3f}" if r2 else "—")
    c2.metric("RMSE", f"{rmse:.3f} t/ha" if rmse else "—")
    if mae:
        st.caption(f"Mean absolute error **{mae:.3f} t/ha** "
                   f"(≈ {mae * 10 * float(price_clean['price_birr_per_quintal'].mean()):,.0f} birr/ha).")

    st.markdown("### 🔍 How it works")
    st.markdown(
        f"""
        1. You pick **region, crop, year, planting month** and the plot measurements.
        2. The app builds the plot's 4-month growing-season window and aggregates the
           regional weather table onto it automatically.
        3. The trained model (**{selected_model}**, chosen from
           **{card.get('n_combinations_tried', 108):,}** model × encoder × scaler combinations)
           predicts yield.
        4. Price for that region/crop/year is looked up and turned into revenue.
        """
    )
    st.markdown("### 📚 Data")
    st.caption(
        "15,090 training plots · 3,750 test plots · 5 regions · 5 crops · 2021–2024, "
        "joined with regional monthly weather and market prices."
    )
    st.info("Weather and price are **never** typed in — they are looked up from bundled cleaned tables.",
            icon=":material/cloud_done:")


REGION_CLIM = build_region_temp_climatology(weather_clean)


def build_row(region, crop_type, survey_year, planting_month, **plot):
    """Assemble one complete MODEL_FEATURES row, looking weather up automatically.

    Uses the *same* ``add_engineered_features`` function notebook 01 uses, so the
    demo and the training pipeline can never disagree about a feature's definition.
    """
    season = season_features_for_one_plot(weather_clean, region, int(survey_year), planting_month)
    row = {
        "region": region,
        "crop_type": crop_type,
        "planting_month": planting_month,
        "survey_year": int(survey_year),
        **plot,
        **season,
    }
    frame = add_engineered_features(pd.DataFrame([row]), REGION_CLIM, CROP_OPTIMA)
    frame["temp_deviation_from_region_avg"] = frame["temp_deviation_from_region_avg"].fillna(0.0)
    return frame.iloc[0].to_dict(), season


def predict_yield(**kwargs):
    row, season = build_row(**kwargs)
    frame = pd.DataFrame([row])[MODEL_FEATURES]
    return max(float(model.predict(frame)[0]), 0.0), season


tab_predict, tab_explore, tab_about = st.tabs(
    ["🔮  Predict a plot", "📊  Explore the data", "ℹ️  About"])

# ---------------------------------------------------------------- predictor ---
with tab_predict:
    with st.form("plot_form"):
        st.markdown("#### 1 · Where and what")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            region = st.selectbox("Region", CANONICAL_REGIONS, format_func=str.capitalize)
        with c2:
            crop_type = st.selectbox("Crop", CANONICAL_CROPS, format_func=str.capitalize)
        with c3:
            survey_year = st.number_input("Survey year", min_value=2021, max_value=2024, value=2024, step=1)
        with c4:
            planting_month = st.selectbox("Planting month", MONTH_ORDER, format_func=str.capitalize)

        st.markdown("#### 2 · Plot measurements")
        p1, p2, p3 = st.columns(3)
        with p1:
            altitude_m = st.slider("Altitude (m above sea level)", 0.0, 3500.0, 1800.0, 25.0)
            farm_size_ha = st.number_input("Farm size (ha)", min_value=0.05, max_value=100.0, value=1.0, step=0.05)
            fertilizer_kg_per_ha = st.slider("Fertilizer applied (kg/ha)", 0.0, 250.0, 60.0, 1.0)
        with p2:
            soil_quality_index = st.slider("Soil quality index (0–1)", 0.0, 1.0, 0.55, 0.01)
            labor_days_per_ha = st.slider("Labour (days/ha)", 0.0, 150.0, 45.0, 1.0)
            distance_to_market_km = st.slider("Distance to market (km)", 0.0, 250.0, 12.0, 1.0)
        with p3:
            improved_seed_used = st.checkbox("Improved seed used", value=True)
            pest_disease_flag = st.checkbox("Pest / disease pressure observed", value=False)
            st.caption("Season rainfall is taken from your plot's own survey estimate; the "
                       "station-based season totals are computed for you.")

        submitted = st.form_submit_button("🌾  Predict yield and revenue", use_container_width=True)

    if submitted:
        if farm_size_ha <= 0:
            st.warning("Farm size must be greater than zero — please enter a positive area.",
                       icon=":material/warning:")
            st.stop()
        if altitude_m > 3200:
            st.info("Above roughly 3,200 m almost no crop is grown in this dataset; the prediction "
                    "will be extrapolated.", icon=":material/terrain:")
        if fertilizer_kg_per_ha > 200:
            st.info("Rates above ~200 kg/ha are far outside the training range; treat the result "
                    "as an extrapolation.", icon=":material/science:")

        try:
            plot = dict(
                altitude_m=float(altitude_m),
                rainfall_mm_season=None,  # filled below from the looked-up season total
                farm_size_ha=float(farm_size_ha),
                fertilizer_kg_per_ha=float(fertilizer_kg_per_ha),
                improved_seed_used=int(improved_seed_used),
                pest_disease_flag=int(pest_disease_flag),
                soil_quality_index=float(soil_quality_index),
                labor_days_per_ha=float(labor_days_per_ha),
                distance_to_market_km=float(distance_to_market_km),
            )
            row, season = build_row(region, crop_type, int(survey_year), planting_month, **plot)
            # the survey's own rainfall column is filled with the looked-up station
            # season total, because the demo must never ask the user for a weather value
            row["rainfall_mm_season"] = row["season_rainfall_mm_total"]
            predicted_yield = max(float(model.predict(pd.DataFrame([row])[MODEL_FEATURES])[0]), 0.0)
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Could not generate a prediction: {exc}", icon=":material/error:")
            st.stop()

        price_match = price_clean[
            (price_clean["region"] == region)
            & (price_clean["crop_type"] == crop_type)
            & (price_clean["year"] == int(survey_year))
        ]
        if not price_match.empty:
            price = float(price_match["price_birr_per_quintal"].iloc[0])
            price_note = (f"exact match for {crop_type.capitalize()} in {region.capitalize()}, "
                          f"{int(survey_year)}")
        else:
            crop_prices = price_clean.loc[price_clean["crop_type"] == crop_type, "price_birr_per_quintal"]
            price = float(crop_prices.mean()) if len(crop_prices) else 0.0
            price_note = f"no exact region/year match — {crop_type.capitalize()} overall average"

        revenue = predicted_yield * farm_size_ha * 10 * price

        avg_row = region_crop[(region_crop["region"] == region) & (region_crop["crop_type"] == crop_type)]
        region_crop_avg = float(avg_row["mean_yield"].iloc[0]) if not avg_row.empty else 0.0
        n_plots = int(avg_row["n_plots"].iloc[0]) if not avg_row.empty else 0
        overall_avg = float(region_crop["mean_yield"].mean()) if not region_crop.empty else 0.0

        st.markdown("### 📈 Result")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f'<div class="kpi"><div class="label">Predicted yield</div>'
                f'<div class="value">{predicted_yield:.2f}</div>'
                f'<div class="sub">tonnes per hectare</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(
                f'<div class="kpi amber"><div class="label">Estimated revenue</div>'
                f'<div class="value">{revenue:,.0f}</div>'
                f'<div class="sub">birr for this {farm_size_ha:.2f} ha plot</div></div>',
                unsafe_allow_html=True)
        with k3:
            delta = predicted_yield - region_crop_avg
            sign = "+" if delta >= 0 else ""
            st.markdown(
                f'<div class="kpi"><div class="label">vs {region.capitalize()} {crop_type} average</div>'
                f'<div class="value">{sign}{delta:.2f}</div>'
                f'<div class="sub">t/ha · average is {region_crop_avg:.2f}</div></div>',
                unsafe_allow_html=True)
        with k4:
            st.markdown(
                f'<div class="kpi"><div class="label">Price used</div>'
                f'<div class="value">{price:,.0f}</div>'
                f'<div class="sub">birr per quintal</div></div>', unsafe_allow_html=True)

        st.markdown("#### What was looked up for you automatically")
        months_note = (f"{season['season_months_matched']}/4 months matched"
                       if season["season_months_matched"] > 0
                       else "no exact match — used regional climate averages")
        st.markdown(
            f"""<div class="lookup">
            ☁️ <b>Growing season</b> ({planting_month.capitalize()} {int(survey_year)} + 3 months, {months_note}):
            average <b>{season['season_avg_temp_c']:.1f} °C</b>,
            total rainfall <b>{season['season_rainfall_mm_total']:.0f} mm</b>,
            <b>{season['season_extreme_heat_days']:.0f}</b> extreme-heat days.<br>
            💰 <b>Price</b>: <b>{price:,.0f} birr/quintal</b> — {price_note}.<br>
            🧮 <b>Revenue</b> = {predicted_yield:.2f} t/ha × {farm_size_ha:.2f} ha × 10 quintals/t
            × {price:,.0f} birr/quintal = <b>{revenue:,.0f} birr</b>.
            </div>""",
            unsafe_allow_html=True,
        )

        chart_row = dict(
            altitude_m=float(altitude_m), farm_size_ha=float(farm_size_ha),
            fertilizer_kg_per_ha=float(fertilizer_kg_per_ha),
            improved_seed_used=int(improved_seed_used), pest_disease_flag=int(pest_disease_flag),
            soil_quality_index=float(soil_quality_index), labor_days_per_ha=float(labor_days_per_ha),
            distance_to_market_km=float(distance_to_market_km),
        )

        def predict_with_override(**overrides):
            p = {**chart_row, **overrides}
            r, _s = build_row(region, crop_type, int(survey_year), planting_month, **p)
            r["rainfall_mm_season"] = r["season_rainfall_mm_total"]
            return max(float(model.predict(pd.DataFrame([r])[MODEL_FEATURES])[0]), 0.0)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Your plot", f"{region.capitalize()} {crop_type} avg", "National avg"],
                y=[predicted_yield, region_crop_avg, overall_avg],
                marker_color=[GREEN, "#3b82f6", "#94a3b8"],
                text=[f"{v:.2f}" for v in [predicted_yield, region_crop_avg, overall_avg]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Your prediction vs history ({n_plots:,} similar plots)",
                yaxis_title="Yield (t/ha)", showlegend=False, height=380,
                plot_bgcolor="white", margin=dict(t=60, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fert_grid = np.linspace(0, min(250, max(150, fertilizer_kg_per_ha * 1.5)), 28)
            curve = [predict_with_override(fertilizer_kg_per_ha=float(f)) for f in fert_grid]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fert_grid, y=curve, mode="lines",
                                     line=dict(color=GREEN, width=3), name="Yield"))
            fig.add_trace(go.Scatter(x=[fertilizer_kg_per_ha], y=[predicted_yield], mode="markers",
                                     marker=dict(size=13, color=AMBER, line=dict(color=GREEN_DARK, width=2)),
                                     name="Your rate"))
            fig.update_layout(
                title="What if: fertilizer response for this plot",
                xaxis_title="Fertilizer (kg/ha)", yaxis_title="Predicted yield (t/ha)",
                height=380, plot_bgcolor="white", margin=dict(t=60, b=10, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.6),
            )
            st.plotly_chart(fig, use_container_width=True)

        month_grid = list(MONTH_ORDER)
        month_vals = []
        for m in month_grid:
            r, _s = build_row(region, crop_type, int(survey_year), m, **chart_row)
            r["rainfall_mm_season"] = r["season_rainfall_mm_total"]
            month_vals.append(max(float(model.predict(pd.DataFrame([r])[MODEL_FEATURES])[0]), 0.0))
        best_month = month_grid[int(np.argmax(month_vals))]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=[m.capitalize() for m in month_grid], y=month_vals,
                             marker_color=[AMBER if m == planting_month else GREEN_LIGHT
                                           for m in month_grid],
                             text=[f"{v:.2f}" for v in month_vals], textposition="outside"))
        fig.update_layout(
            title=f"Planting-month sensitivity (your month highlighted · best is {best_month.capitalize()})",
            xaxis_title="Planting month", yaxis_title="Predicted yield (t/ha)", showlegend=False,
            height=360, plot_bgcolor="white", margin=dict(t=60, b=10, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Model: **{selected_model}** · validation R² {r2:.3f}"
            + (f" · 5-fold CV RMSE {card['cv_rmse_mean']:.3f} ± {card['cv_rmse_std']:.3f} t/ha"
               if card.get("cv_rmse_mean") else "")
            + " · weather features improve validation RMSE by "
            + f"{card.get('weather_ablation_rmse_gain', 0):.3f} t/ha."
        )

# ------------------------------------------------------------------ explore ---
with tab_explore:
    st.markdown("#### Explore the cleaned historical data")
    f1, f2, f3 = st.columns(3)
    with f1:
        regions_sel = st.multiselect("Regions", CANONICAL_REGIONS, default=CANONICAL_REGIONS,
                                     format_func=str.capitalize)
    with f2:
        crops_sel = st.multiselect("Crops", CANONICAL_CROPS, default=CANONICAL_CROPS,
                                   format_func=str.capitalize)
    with f3:
        years_sel = st.multiselect("Years", [2021, 2022, 2023, 2024], default=[2021, 2022, 2023, 2024])

    view = region_crop[region_crop["region"].isin(regions_sel) & region_crop["crop_type"].isin(crops_sel)]
    if view.empty:
        st.warning("No combinations match that filter — pick at least one region and one crop.",
                   icon=":material/filter_alt_off:")
    else:
        heat = view.pivot_table(index="region", columns="crop_type", values="mean_yield")
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=[c.capitalize() for c in heat.columns], y=[r.capitalize() for r in heat.index],
            colorscale="YlGn", colorbar=dict(title="t/ha"),
            text=np.round(heat.values, 2), texttemplate="%{text}", hovertemplate="%{y} · %{x}<br>%{z:.2f} t/ha<extra></extra>",
        ))
        fig.update_layout(title="Mean yield by region × crop (t/ha)", height=420,
                          plot_bgcolor="white", margin=dict(t=60, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        bars = (region_crop[region_crop["crop_type"].isin(crops_sel)]
                .groupby("crop_type", as_index=False)["mean_yield"].mean()
                .sort_values("mean_yield", ascending=False))
        fig = go.Figure(go.Bar(x=[c.capitalize() for c in bars["crop_type"]], y=bars["mean_yield"],
                               marker_color=GREEN, text=np.round(bars["mean_yield"], 2),
                               textposition="outside"))
        fig.update_layout(title="Average yield by crop across the selected regions",
                          yaxis_title="Yield (t/ha)", height=380, showlegend=False,
                          plot_bgcolor="white", margin=dict(t=60, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### Price trend for the selected crops")
        price_view = price_clean[price_clean["crop_type"].isin(crops_sel)]
        price_year = price_view.groupby(["year", "crop_type"], as_index=False)["price_birr_per_quintal"].mean()
        if not price_year.empty:
            fig = go.Figure()
            for crop in sorted(price_year["crop_type"].unique()):
                sub = price_year[price_year["crop_type"] == crop]
                fig.add_trace(go.Scatter(x=sub["year"], y=sub["price_birr_per_quintal"], mode="lines+markers",
                                         name=crop.capitalize()))
            fig.update_layout(title="Average price per quintal by year (unit-corrected)",
                              xaxis_title="Year", yaxis_title="birr / quintal", height=380,
                              plot_bgcolor="white", margin=dict(t=60, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Region/crop means come from the cleaned training table; prices from the cleaned, "
                   "unit-corrected market price table.")

# -------------------------------------------------------------------- about ---
with tab_about:
    st.markdown(
        f"""
        ### About this project

        **Developed by {AUTHOR}** for the Qiyas AI Hackathon 2026 — **Ethiopian Crop Yield & Revenue Predictor**.

        The project predicts `yield_tons_per_ha` for 3,750 unseen plots from plot survey data joined with
        regional monthly weather and market prices, then converts the prediction into an estimated revenue.

        **Pipeline**
        - Three raw exports are cleaned in code (inconsistent region/crop spellings, two different missing-value
          sentinels, duplicate readings, a price unit mix-up) and the growing-season weather window
          (planting month + 3) is joined onto every plot as a many-to-one join.
        - **{len(MODEL_FEATURES)} model features**: {len(CATEGORICAL_FEATURES) + len(PLOT_NUMERIC_FEATURES)}
          from the cleaned plot survey, {len(WEATHER_FEATURES)} weather-derived features, and
          {len(ENGINEERED_FEATURES)} engineered threshold/interaction/crop-optimum features.
        - **{card.get('n_candidate_models', 18)} models × 2 encoders × 3 scalers =
          {card.get('n_combinations_tried', 108)} combinations** are scored on one shared validation split;
          the winner (**{selected_model}**) is tuned, optionally blended with the runner-up families, and
          refitted on 100% of the training data.
        - Validation R² **{f"{r2:.3f}" if r2 else "—"}**, RMSE **{f"{rmse:.3f} t/ha" if rmse else "—"}**.

        **Reproducing everything**

        ```bash
        pip install -r requirements.txt
        jupyter lab            # run notebooks 01 → 02 → 03 → 04
        streamlit run app/app.py
        ```

        Every number shown in this app is produced by code in this repository — the weather and price
        lookups come from the cleaned tables in `app/assets/`.
        """
    )

st.markdown(
    f'<div class="footer"><b>Ethiopian Crop Yield &amp; Revenue Predictor</b> · Built by <b>{AUTHOR}</b> · '
    f'Qiyas AI Hackathon 2026 · prediction = yield × price, never a hand-typed weather value.</div>',
    unsafe_allow_html=True,
)
