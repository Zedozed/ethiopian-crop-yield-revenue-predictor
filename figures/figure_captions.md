# Figure Captions

One or two sentences per figure: what it shows, and the takeaway ("so what").

- **fig01_missingness.png:** Shows percent missing (including -999 sentinels) across all three raw tables; fertilizer_kg_per_ha, soil_quality_index, labor_days_per_ha are the most affected columns in the plot file, which is why they get train-fit median/mode imputation in deliverable A.
- **fig02_before_after_cleaning.png:** Compares fertilizer_kg_per_ha and price_birr_per_quintal before and after cleaning; capping fertilizer at the train 99th percentile and multiplying the mispriced rows by 100 both fix the problem without distorting the rest of each distribution.
- **fig03_yield_distribution.png:** Yield is right-skewed overall (mean 2.76 tons/ha); the by-crop panel shows that skew is not uniform -- some crops have a wider spread than others.
- **fig04_region_crop_heatmap.png:** Mean yield varies sharply by region -- the hot Somali region sits far below the others -- and by crop within each region; the darkest cell marks the single best-performing region-crop combination, and the (n=...) labels show which cells have enough plots to trust.
- **fig05_correlation_heatmap.png:** The strongest single correlate of yield is the crop-specific temperature gap temp_dev_from_crop_optimum_c (-0.47), followed by the heat-stress and season-temperature features (all clearly negative). Even the best single feature is only moderately correlated, which is why D uses boosted trees over plain linear regression.
- **fig06_climate_by_region.png:** Temperature follows a clear seasonal curve that differs by region; the shaded band shows one example growing-season window, illustrating why the join aggregates weather per region and per season rather than using a single national number.
- **fig07_yield_vs_season_temp.png:** Each crop's binned-mean trend line shows its own relationship with season temperature -- some crops have a visible sweet spot, others are closer to flat across the observed range.
- **fig08_price_trends.png:** Crop prices generally trended upward from 2021 to 2024, with teff consistently the most expensive crop per quintal.
- **fig09_revenue_by_crop_region.png:** Revenue per hectare reflects both yield and price, so the highest-yield crop is not always the highest-revenue crop once price is factored in.
- **fig10_model_comparison.png:** Validation RMSE for all 18 model families, each at its best of the 108 model x encoder x scaler combinations, against the mean-predictor baseline. catboost_log_target wins at 0.423 t/ha; the error bar is its 5-fold CV spread.
- **fig11_predicted_vs_actual_residuals.png:** Predicted vs actual yield clusters tightly around the y = x line with residuals centred on zero (mean +0.024 t/ha), so the model is well calibrated; the slight funnel at high predicted yield shows a little more error on the best plots.
- **fig12_feature_importance.png:** crop_type is the most important feature by permutation importance, and 5 of the top 14 features are weather-derived (red) -- confirming the growing-season weather join, not the plot survey, carries most of the signal.
