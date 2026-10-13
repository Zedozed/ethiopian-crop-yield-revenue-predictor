# Notebook Guide — What Every Part of This Project Is Actually Doing

**Developed by Zedingil Zelalem** · Qiyas AI Hackathon 2026

Plain-language walkthrough of every notebook (cell by cell, in run order), every function
in `src/cleaning.py`, `src/features.py` and `src/train.py`, and every section of the demo
app (`app/app.py`). Written so you (or a teammate, or a grader skimming quickly) can
understand what a piece of code is for without reading the code itself first.

---

## Notebook 01 — `01_cleaning_and_integration.ipynb`

**Job of this whole notebook:** take the 3 messy raw CSVs and turn them into two clean,
model-ready tables (`master_train.csv`, `master_test.csv`), while proving every cleaning
decision was reasonable.

1. **Load the raw files** — Reads the 3 original CSVs exactly as downloaded, with zero
   edits. Everything from here on works off these in-memory copies, so the raw files on
   disk are never touched.

2. **A1 — Cataloguing every data problem we found** — Doesn't fix anything yet. Just
   counts and lists every issue in the raw data (inconsistent spelling, sentinel missing
   values, outliers, a price unit mix-up) into one table, with the fix we're about to
   apply and why. This is the "evidence" cell — it proves the cleaning wasn't guesswork.

3. **A2 — Actually cleaning the labels, with before/after proof** — This is where
   cleaning really happens: region and crop names get standardized, two different missing
   -value codes get converted to real NaNs, and the test set gets the exact same
   statistics (medians, caps) that were learned from train — never its own. Prints the
   unique values before and after so you can see the mess disappear.

4. **A3 — Drawing the join map** — A picture (saved to `reports/join_diagram.png`)
   showing which table is on which side of each join, and why. No calculation here, just
   the visual explanation of the plan.

5. **A4 — Checking the join actually worked** — Runs the real join and reports hard
   numbers: what percent of plots got a full 4-month weather match vs. a partial or no
   match, and what percent matched a price row. This is the honesty check — if match
   rates were terrible, this cell would be where you'd find out.

6. **A5 — Showing the join's work for 3 real plots** — Picks 3 specific plots and prints
   exactly which weather rows got pulled in for each one and the resulting season
   averages, so the join logic is auditable by hand, not just trusted as a black box.

7. **A6 — Building the engineered features and explaining each one** — Creates the 22
   engineered columns and prints a table explaining each one's formula, source columns
   and why it should help. Three groups:
   - *Season-window weather statistics* (15) — mean / max / min / spread of season
     temperature, total / wettest / driest month rainfall, total and peak extreme-heat
     days, the planting-month and pre-planting-month readings, and the season-months-
     matched join-quality flag.
   - *Explicit stress thresholds* (3) — `temp_excess_above_threshold_c`,
     `heat_stress_days_excess`, `season_rain_deficit_mm`. Notebook 02/03 show yield is
     roughly flat up to ~21 °C and ~220 mm of season rain and then moves sharply; these
     make those breakpoints explicit instead of hoping the model rediscovers them.
   - *Interactions and timing* (4) — `fertilizer_x_improved_seed`, `soil_x_fertilizer`,
     `rainfall_x_temp`, and the numeric `planting_month_num`.

8. **A7 — The integrity checks (pass/fail)** — 13 sanity asserts: no duplicate plot IDs,
   no missing values left in model features (train *and* test), region/crop values all
   within the expected set, row counts unchanged by the joins, non-negative rainfall,
   season-months-matched between 0 and 4, strictly positive cleaned prices, and train
   and test sharing identical feature columns. If anything here prints `FAIL`, stop and
   fix it before trusting anything downstream.

9. **A8 — Saving everything downstream notebooks need** — Writes `master_train.csv`,
   `master_test.csv`, the data dictionary (one row per column with its type, source file
   and how it was derived), the fitted cleaning statistics (so nothing downstream has to
   re-derive them), and copies the cleaned weather/price tables plus a compact
   `region_crop_summary.csv` into `app/assets/` for the demo. Nothing after this cell
   changes any data — it's all reads from here on.

---

## Notebook 02 — `02_analysis_report.ipynb`

**Job of this whole notebook:** answer 14 specific analysis questions about the cleaned
data, each with a number/table/chart and a one-sentence takeaway computed directly from
the data (not a number I guessed).

- **Setup** — Loads the master table plus the cleaned weather/price tables.
- **B1.1 / B1.2 — Money questions** — Revenue per hectare by crop, and how crop prices
  moved 2021→2024.
- **B2.1 – B2.5 — Yield questions** — Yield by region, by crop, the region×crop
  combination table, how much improved seed actually helps per crop, and how much pest/
  disease pressure costs per crop.
- **B3.1 – B3.4 — "What drives yield" questions** — Yield vs. fertilizer amount, yield
  vs. altitude (flagging bins with too few plots to trust), yield vs. planting month, and
  whether distance to market correlates with yield at all.
- **B4.1 – B4.3 — Time and data-quality questions** — Yield trend across survey years,
  which region-year combos ran unusually hot, and whether a farmer's own rainfall
  estimate agrees with the station-measured rainfall (a data-trust check, not just an
  analysis one).

Every one of these cells prints its own interpretation sentence computed from whatever
the data actually says — that's deliberate, so the numbers in your report always match
what's really in `master_train.csv`, even if you re-run after a cleaning change.

What this notebook establishes, and why the model in notebook 04 looks the way it does:
region matters enormously (the hot **Somali** region averages 1.46 t/ha against ~3.0–3.1
t/ha everywhere else), season temperature and extreme-heat days are the strongest single
correlates of yield (both negative, ≈ −0.45), season rainfall helps only up to a
saturation point, and the plot-level inputs (soil quality, fertilizer, improved seed,
labour) shift yield multiplicatively rather than additively.

---

## Notebook 03 — `03_visualizations.ipynb`

**Job of this whole notebook:** produce figures 1–9 (figures 10–12 need the trained
model, so they're made at the end of notebook 04 instead) and auto-write their captions.

- **Setup** — Loads the raw train file (to show "before"), the master table ("after"),
  and the cleaned weather/price tables.
- **Figure 1 — Where the missing data was** — A bar chart of percent-missing per raw
  column, including the `-999` sentinels, before anything got cleaned.
- **Figure 2 — Fertilizer, before vs. after cleaning** — Two histograms side by side
  showing the outlier cap doing its job without distorting the rest of the distribution.
- **Figure 3 — Overall yield distribution** — A plain histogram of the target variable,
  so you know its shape before judging any model's error against it.
- **Figure 4 — Yield heatmap by region and crop** — The single most "where should I
  farm what" figure in the whole project.
- **Figure 5 — Correlation heatmap** — Now covers the plot survey columns, all 15
  weather-derived features and the engineered interactions. It shows the strongest
  correlations with yield are the season temperature / extreme-heat counts, followed by
  season rainfall and the soil/fertilizer features — all moderate, which is the
  justification for a non-linear model.
- **Figure 6 — Climate differences by region** — Monthly temperature curves per region,
  justifying why weather was joined per-region and per-season instead of nationally.
- **Figure 7 — Yield vs. season temperature** — One panel per crop, with binned means,
  showing the sweet spot / collapse pattern that the threshold features encode.
- **Figure 8 — Price trends** — Same price-over-time data as B1.2, saved as a standalone
  figure for the figures/ folder requirement.
- **Figure 9 — Revenue by crop and region** — Combines yield and price into one view,
  showing that "highest yield" and "highest revenue" aren't always the same crop.
- **Caption-writing cell** — Reads `figure_captions.md`, fills in the 9 captions this
  notebook owns (computed from the real numbers above, not hand-written guesses), and
  leaves the other 3 lines untouched for notebook 04 to fill in.

---

## Notebook 04 — `04_modeling_and_evaluation.ipynb`

**Job of this whole notebook:** find the best model out of a large, explicit search,
prove it's actually good (not just lucky), explain where it still struggles, and produce
the real submission file.

**The selection design.** Every candidate is a single scikit-learn `Pipeline` combining a
**categorical encoder**, a **numeric scaler**, and a regressor — so the exact fitted steps
run on train, test, and in the live demo. The grid is
`18 models × 2 encoders × 3 scalers = 108 combinations`:

- *Models*: mean predictor, linear regression, ridge, lasso, elastic net, KNN, decision
  tree, random forest, extra trees, gradient boosting, histogram gradient boosting, MLP,
  SVR, XGBoost, LightGBM, CatBoost, and log-target variants of LightGBM and CatBoost.
- *Encoders*: `OneHotEncoder`, `OrdinalEncoder`.
- *Scalers*: `StandardScaler`, `MinMaxScaler`, `RobustScaler`.

- **Setup** — Loads the master tables, imports the feature contract from
  `src/features.py` and the model zoo from `src/train.py`, and splits off a fixed
  validation set (20% of the train file, `random_state=42`).
- **D1 — How much does a "dumb" model get wrong?** — A mean-only predictor and plain
  linear regression, to set the floor any real model must beat.
- **D2 — The full bake-off** — Two passes. The *screening* pass trains all 108
  combinations on a stratified 4,000-row subsample of the training split and scores them
  on the full validation split. The *confirmation* pass refits each family's best
  encoder/scaler combo on the **full** training split and re-scores it. The ranking in
  the table is the confirmation pass; the whole table is exported to
  `reports/D2_model_selection.csv`. A useful side finding the table makes visible: the
  encoder and scaler barely move the tree/boosting models (they're invariant to monotone
  rescaling) but matter a lot for KNN, the SVR and the MLP.
- **D3 — Is the winner just lucky?** — 5-fold cross-validation for the winner and the
  runner-up. A small standard deviation means "no, it's consistently good," not a fluke
  of one lucky split.
- **D4 — Does it still work on data from the future?** — Trains only on 2021–2023 and
  validates on 2024, to catch a model that's secretly memorising year-specific patterns.
- **D5 — Do the weather features we worked hard to build actually help?** — Re-trains the
  same model with all 19 weather-derived features removed (`WEATHER_FEATURES` in
  `src/features.py`), to put a real number on whether that join effort was worth it.
- **D6 — Squeezing more accuracy out of the winner** — A `RandomizedSearchCV` over the
  winner's documented search space (`TUNING_SPACES` in `src/train.py`), scored by 3-fold
  cross-validated RMSE on the training split only, reporting search space, trial count,
  best parameters and before/after scores.
- **D6.5 — Does averaging models beat one?** — Builds an equal-weight blend of the tuned
  winner with the next-best *distinct* families and only switches to it if it genuinely
  beats the single model on validation RMSE. Either way the cell tells you which one won
  and by how much.
- **D7 — Where does the model still get it wrong?** — (a) error broken down by crop and
  by region, in tons/ha *and* as a percentage of each region's own yield; (b) error
  against continuous features (season temperature, fertilizer rate), looking for
  worse-at-the-extremes behaviour; (c) the 10 worst validation plots listed with
  hypotheses for why those groups are hard.
- **D8 — So what did we do about it?** — A direct, data-driven response to D7, including
  the feature additions that came out of it and an honest note about the region-specific
  correction that was tried and did *not* help.
- **D9 — Translating the error into plain language** — Converts RMSE/MAE from "tons per
  hectare" into "percent of a typical yield" and "birr of revenue error per hectare" —
  the number a non-technical judge or farmer would actually understand.
- **Final model, predictions, submission** — Refits the chosen model (single or blend) on
  100% of the training data, saves it to `models/final_model.joblib`, writes a machine-
  readable model card to `reports/final_model_metrics.json` (which the demo app reads to
  display the real R²/RMSE), predicts on the real leaderboard test set, and writes
  `submission/zedingil_zelalem_submission.csv` in the template's original plot_id
  order, with asserts on row count, uniqueness, ordering and non-null predictions.
- **Figures 10, 11, 12** — Validation RMSE of every model family (with the mean-predictor
  baseline marked and a CV error bar on the winner), predicted-vs-actual + residual
  plots, and permutation feature importance with the weather-derived features highlighted.
- **Final caption-writing cell** — Fills in the last 3 lines of `figure_captions.md`, so
  after this notebook runs, all 12 figures have real captions.

---

## `src/cleaning.py` — shared cleaning functions

**Job of this file:** every cleaning rule that notebook 01 uses, written once as plain
functions so the exact same logic can be reused and never silently drifts between train
and test.

- **`CANONICAL_REGIONS`, `CANONICAL_CROPS`** — the single list of "correct" spellings
  (`amhara, oromia, snnpr, somali, tigray` and the 5 crop names) that every table gets
  normalized to. If a grader asks "what's the allowed set of labels," this is it.
- **`MONTH_ORDER`, `MONTH_NUM`** — the 12 month abbreviations (`jan`...`dec`) and their
  1–12 numeric position. Used anywhere a month needs to become a number, or a number
  needs to become the next/previous month (season-window math).
- **`_REGION_ABBREV`, `_CROP_FIXES`** — the two small lookup dicts that do the actual
  relabeling: `{"amh": "amhara", "oro": "oromia", ...}` for the weather file's
  abbreviations, and `{"tef": "teff"}` for the one misspelling found in the plot file.
- **`_IMPUTE_GROUP_COL`** — for each column that had missing values, which other column
  to group by when filling them in: fertilizer and labor by `crop_type` (they vary a lot
  by crop), rainfall by `region` (it's a climate effect, not a crop effect), soil quality
  by nothing (just the overall median) since it's plot-specific noise.
- **`_LOW_PRICE_THRESHOLD = 500`** — the cutoff used to spot the birr-per-kg vs.
  birr-per-quintal mix-up: any price below 500 is assumed to be in the wrong unit and
  gets multiplied by 100.
- **`_norm_region(series)` / `_norm_crop(series)`** — tiny helpers that strip whitespace,
  lowercase, and apply the abbreviation/misspelling fix for one column at a time.
- **`clean_plot_table(df, stats=None)`** — the most important function here. Call it once
  with `stats=None` on the **train** file: it cleans the labels, converts the two
  missing-value sentinels (`-999` and `-999.0`) to real NaNs, computes the fertilizer
  outlier cap and the imputation medians/mode *from this data*, fills everything in, and
  **returns both the cleaned table and those computed stats**. Call it again with
  `stats=<the dict you just got back>` on the **test** file: this time it skips
  recomputing anything and just applies the same train-derived numbers. That's the whole
  mechanism behind "fit on train only" (Rule 6) — one function, two modes, driven by
  whether you pass `stats`.
- **`clean_weather_table(df)`** — cleans region labels, drops the exact duplicate rows,
  averages the conflicting duplicate region-year-month readings, and fills any remaining
  blank weather cells with that region+month's own across-year average. No train/test
  split here — weather is a shared reference table, not per-plot data, so there's nothing
  to leak.
- **`clean_price_table(df)`** — cleans labels, fixes the unit mix-up, and fills blank
  prices with that crop's median price. Same reasoning as weather: a shared lookup table.
- **`save_stats(stats, path)` / `load_stats(path)`** — just read/write the fitted-stats
  dict as JSON, so `cleaning_stats.json` is a plain, inspectable file rather than a
  pickle.

## `src/features.py` — the season-window join and the model's feature contract

**Job of this file:** turn cleaned weather into per-plot season features, and define, in
exactly one place, which columns the model is trained and predicted on — so training
(notebook 04) and the live demo (`app.py`) can never disagree about what a "feature" is.

- **`CATEGORICAL_FEATURES`, `PLOT_NUMERIC_FEATURES`, `WEATHER_FEATURES`,
  `ENGINEERED_FEATURES`, `NUMERIC_FEATURES`, `MODEL_FEATURES`, `TARGET_COLUMN`** — the
  feature contract. `MODEL_FEATURES` is just the lists concatenated; everything downstream
  (the training pipeline, permutation importance, the demo's input row) builds its column
  list from these constants instead of retyping it. `WEATHER_FEATURES` doubles as the
  exact set notebook 04's D5 ablation removes.
- **`SEASON_LENGTH = 4`** — the growing-season window length: planting month plus the 3
  months after it.
- **`GRAIN_HEAT_THRESHOLD_C = 21.0`, `SEASON_RAIN_SUFFICIENT_MM = 220.0`** — the two
  non-linear breakpoints found in the analysis, kept as named constants so the threshold
  features and the demo app quote the same numbers.
- **`season_window_months(planting_month, year, n=4, start_offset=0)`** — given a start
  month and year, returns the list of `(month, year)` pairs the season covers, correctly
  rolling over December→January (and back, for `start_offset=-1`, which gives the
  pre-planting month). This is the one place that wraparound logic lives.
- **`build_weather_lookup(weather_clean)`** — indexes the weather table by
  `(region, year, month)` once, so the per-plot join doesn't have to brute-force scan the
  frame 15,090 times.
- **`aggregate_season_window(lookup, region, year, planting_month)`** — the single source
  of truth for "what was the weather during this plot's growing season." Returns the 14
  season/planting-month/pre-planting statistics plus `season_months_matched`. Both the
  batch join in notebook 01 and the single-plot lookup in `app.py` go through this, which
  is the guarantee that "growing season" means the same thing in both places.
- **`aggregate_season_weather(weather_clean, region, year, planting_month)`** — a thin
  convenience wrapper around the two functions above, used by notebook 01's A4 join audit
  and A5 join proof.
- **`build_region_temp_climatology` / `build_region_rain_climatology`** — each region's
  average temperature and 4-month rainfall across every month and year on record. Used as
  the baseline that `temp_deviation_from_region_avg` measures against, and as the fallback
  when a plot's exact season window has no matching weather rows at all.
- **`compute_crop_optima(joined_train)`** — fits the two *crop-optimum* constants (each
  crop's yield-weighted mean altitude and yield-weighted mean growing-season temperature)
  on the training table only. These put every crop's altitude and temperature response on
  one common scale, which is what the altitude/temperature deviation features consume.
- **`save_crop_optima` / `load_crop_optima`** — persist those constants to
  `app/assets/crop_optima.json` so the demo app reuses the exact train-fitted values.
- **`add_engineered_features(df, region_clim, crop_optima)`** — adds every threshold,
  ratio, interaction, altitude/temperature-deviation and `planting_month_num` feature to a
  table that already carries the plot columns and the growing-season weather columns. Both
  notebook 01 and the demo app call this one function, so the training table and a live
  prediction can never disagree about a feature's definition.
- **`build_master_table(plot_clean, weather_clean, crop_optima=None, return_optima=False)`**
  — runs the season aggregation for every plot (this is the actual batch join), attaches the
  results, fills the rare no-weather-match gaps with the region climatology, fits the crop
  optima when none are passed in, then calls `add_engineered_features`. Called with
  `return_optima=True` on the train file and `crop_optima=<train optima>` on the test file,
  which is how the train-only fitting rule (Rule 6) is enforced structurally. This is what
  produces `master_train.csv` / `master_test.csv` in notebook 01's A8 step.
- **`season_features_for_one_plot(...)`** — the demo app's version of the same fallback
  logic for a single plot, so it can run instantly when you click Predict.
- **`attach_price(df, price_clean)`** — joins price onto a *copy* of a table, purely for
  revenue math (notebook 02's analysis, the demo's revenue number). Never called from the
  training feature-building path, which is what physically prevents price from ever
  becoming a model feature rather than just promising it won't.

## `src/train.py` — the model zoo and selection machinery (Deliverable D)

**Job of this file:** hold every candidate model, the encoder/scaler grid, the
screening/confirmation logic, the tuning search spaces, the blend builder, and the
save-the-final-artifacts helper — so notebook 04 reads as a narrative instead of a wall
of estimator imports.

- **`ENCODERS`** — `onehot` (`OneHotEncoder(handle_unknown="ignore")`) and `ordinal`
  (`OrdinalEncoder` with `unknown_value=-1`).
- **`SCALERS`** — `standard`, `minmax`, `robust`.
- **`model_specs()`** — returns the 18-model zoo as `name -> (factory, log_target, note)`.
  XGBoost / LightGBM / CatBoost are imported inside `try/except`, so the notebook still
  runs end to end (with a smaller zoo) in a minimal environment.
- **`build_preprocessor` / `build_pipeline`** — assemble
  `ColumnTransformer([cat encoder, numeric imputer+scaler]) → regressor` into one
  `Pipeline`, optionally wrapped in a `TransformedTargetRegressor(log/exp)` when
  `log_target=True`. That wrapper is how the log-target variants fit `log(yield)` but
  still `predict()` in tons/ha.
- **`rmse` / `score`** — the shared metric helpers (RMSE, MAE, R²).
- **`run_model_selection(...)`** — the 108-combination screening loop; returns a results
  table sorted by RMSE plus every fitted pipeline.
- **`stratified_subsample(...)`** — class-proportional subsample used to screen the grid
  cheaply and traceably rather than fitting 108 models on the full split.
- **`confirm_on_full_split(...)`** — takes each family's best encoder/scaler combo from
  the screening table and refits it on the full training split for the reported ranking.
- **`TUNING_SPACES` / `tune_pipeline(...)`** — documented `RandomizedSearchCV` spaces for
  LightGBM, CatBoost, XGBoost, random forest and histogram gradient boosting. The tuning
  bases are single-threaded so the search's own parallelism doesn't oversubscribe the CPU.
- **`cross_validate(...)`** — 5-fold KFold RMSE, used by D3.
- **`build_voting_ensemble(...)`** — an equal-weight `VotingRegressor` over already-fitted
  member pipelines; the cheap way to test whether averaging beats the single best model.
- **`save_model_artifacts(...)`** — writes `models/final_model.joblib` and the JSON model
  card the README and app read, so those two can never quote a stale score.

## `app/app.py` — the live demo

**Job of this file:** let someone who has never seen the code enter a plot's details and
get back a predicted yield and revenue, with weather and price looked up automatically.

- **Custom CSS theme** — a gradient hero header, KPI cards with coloured left borders, a
  dashed "what was looked up for you" panel and a footer, all injected via one
  `st.markdown(..., unsafe_allow_html=True)` block. The palette is the same green/amber
  farming palette as the figure pack.
- **Hero header + attribution** — states what the tool does in one paragraph and credits
  **Developed by Zedingil Zelalem** both in the hero badge and the footer.
- **`load_assets()`** — loads the trained model (`final_model.joblib`), the cleaned
  weather/price tables, `region_crop_summary.csv` and the train-fitted `crop_optima.json`
  from `app/assets/` (falling back to `data/processed/master_train.csv` if those files
  aren't there yet), all wrapped in
  `@st.cache_resource` so they're read from disk once per session. A
  `try/except FileNotFoundError` shows a friendly message and stops instead of crashing if
  notebooks 01/04 haven't been run yet.
- **`load_model_card()`** — reads `reports/final_model_metrics.json`, so the sidebar's R²,
  RMSE, MAE and chosen-model name are the real numbers produced by notebook 04.
- **Sidebar** — model card, a 4-step "how it works" explainer, the dataset summary, and an
  explicit note that weather and price are never typed in.
- **`build_row(...)`** — assembles one complete `MODEL_FEATURES` row: it calls
  `season_features_for_one_plot` for the looked-up season weather, then runs the same
  threshold, interaction and crop-optimum features through the shared
  `add_engineered_features` function notebook 01 also uses. This
  is the function that keeps the demo's feature math identical to training's.
- **The form** — every input the rubric requires (region, crop, year, planting month,
  altitude, farm size, fertilizer, improved seed, pest/disease, soil quality, labour,
  distance to market), and nothing that counts as a weather number or a price.
- **Friendly input handling** — a zero/negative farm size is refused with a warning;
  altitudes above 3,200 m and fertilizer rates above 200 kg/ha show an "this is an
  extrapolation" notice rather than failing. Any unexpected error surfaces as
  `st.error(...)` instead of a crash.
- **Result KPI cards** — predicted yield (t/ha), estimated revenue for the actual plot
  size (birr), the difference versus that region+crop's historical mean, and the price
  used.
- **The "what was looked up for you" panel** — states the season window, how many of the 4
  months matched, the looked-up average temperature, total rainfall and extreme-heat days,
  the price and exactly how it was found, and the full revenue formula with numbers.
- **Three interactive Plotly charts** — (1) predicted yield vs that region+crop's
  historical average and the national average, (2) a what-if fertilizer response curve for
  this exact plot with your rate marked, and (3) a planting-month sensitivity chart
  highlighting your month and naming the best-performing one.
- **Explore tab** — region/crop/year filters driving three charts: a region × crop yield
  heatmap, average yield by crop, and average price per quintal by year. This is the
  optional "interactive explorer" stretch goal.
- **About tab** — the pipeline in brief, the combination count the winner came from, the
  validation scores and the one-line reproduce commands.

---

## Other project docs worth knowing about

- **`reports/stretch_regional_equity.md`** — the stretch-goal writeup. Shows that the
  worst-served region changes depending on whether you measure error in absolute tons/ha
  or as a percentage of each region's own yield level, and proposes a concrete fix
  (region-calibrated error bands) rather than just flagging the finding.
- **`presentation/zedingil_zelalem_slides.pdf`** — the 5-slide deck (Deliverable F), generated by `src/make_presentation.py` from the figure pack and `reports/final_model_metrics.json`.
- **`app/.streamlit/config.toml`** — the demo's base Streamlit theme (the custom CSS in
  `app.py` layers on top of it).
- **`reports/D2_model_selection.csv`** — the full 108-combination screening table.
- **`reports/final_model_metrics.json`** — the machine-readable model card.
