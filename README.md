<div align="center">

# 🌾 Ethiopian Crop Yield & Revenue Predictor

**Qiyas AI Hackathon 2026 · Solo Submission**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost%20%7C%20log%20target-2E7D32?style=for-the-badge)](#-key-results)
[![Streamlit](https://img.shields.io/badge/Demo-Streamlit%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](#-run-the-demo)
[![JupyterLab](https://img.shields.io/badge/Notebooks-JupyterLab-4B4B8B?style=for-the-badge&logo=jupyter&logoColor=white)](#-quick-start)

**RMSE 0.422 t/ha · MAE 0.304 t/ha · R² 0.911 · shipped in a live Streamlit demo**

</div>

---

Predicts smallholder **crop yield (tons/ha)** across **5 Ethiopian regions and 5 crops** from
plot survey data, regional weather, and market prices — then turns that prediction into an
**estimated revenue**, through a live demo that looks up weather and price automatically.

| Validation | Value |
|:---|---:|
| **RMSE** (held-out 80/20 split) | **0.422 t/ha** |
| **MAE** | **0.304 t/ha** |
| **R²** | **0.911** |
| **5-fold CV** (mean ± std) | 0.427 ± 0.009 |
| **Out-of-time** (2021–23 → 2024) | 0.438 t/ha |
| **Weather ablation gain** | +0.0155 t/ha RMSE when removed |

> **Plain-language score (D9):** a typical prediction is off by ≈ **11% of the plot's yield**, or
> roughly **14,010 birr/ha** at the average 4,616 birr/quintal price — and after the crop-optimum
> features fix, the model's **relative error is nearly uniform across all regions (10.7–11.9%)**.

## ✨ Key results

- **Model sweep:** **18 models** (baselines, linear, KNN, trees, forest, boosting, MLP, SVR —
  plus log-target boosting variants) × **2 encoders** (one-hot, ordinal) × **3 scalers**
  (standard, minmax, robust) = **108 combinations**, screened on a stratified subsample and
  confirmed on the full training split.
- **Best model:** `catboost_log_target`, tuned — one-hot encoder + robust scaler, selected
  from 108 pipelines (see `reports/D2_model_selection.csv`).
- **Crop-optimum features were the biggest jump:** putting every crop's altitude and temperature
  on one common scale lifts the R² from ≈0.89 to **over 0.91** — they rank **#2 and #3 by
  permutation importance** (fig 12), and `temp_dev_from_crop_optimum_c` is the strongest single
  correlate of yield (−0.47).
- **Weather is worth it:** removing the 19 weather-derived features pushes RMSE up by
  **~0.015 t/ha** — the growing-season join measurably improves predictions.
- **Regional equity stretch goal:** after the crop-optimum fix the model's relative error is
  now almost uniform across regions — an earlier version underserved Somali by ~2 points; see
  [`reports/stretch_regional_equity.md`](reports/stretch_regional_equity.md).

## 📚 What happens in each part

| # | Stage | What it does |
|:---:|:---|:---|
| 01 | **Cleaning & integration** | Fixes region/crop label variants (93.9% of rows needed standardising), two missing-value sentinels (−999 / −999.0), a price unit mix-up (birr/kg × 100) and fertilizer outliers capped at the 99th percentile (899 → 136 kg/ha); joins 15,090 plot rows with a 4-month growing-season weather window (80.6% full, 0% unmatched) and market prices — 13/13 integrity checks pass. |
| 02 | **Analysis** | 14 numbered tasks on the cleaned data. Key finding: **Teff has the lowest mean yield (2.00 t/ha) yet the highest revenue (152,301 birr/ha)** — ~1.8× sorghum — while **Somali averages just 1.46 t/ha** vs ~3.0–3.1 elsewhere because season heat sets the level. |
| 03 | **Visualization** | 9 analysis figures (`fig01`–`fig09`) — missingness, before/after cleaning, yield distribution, region×crop heatmap, correlations, climate by region, price trends 2021–24, revenue by crop & region. |
| 04 | **Modeling & evaluation** | The 108-combination sweep, hyperparameter tuning of the winner, 5-fold CV, an out-of-time check, weather ablation, and a full error/equity analysis across crop, region and the hardest plots. |

Then `src/` turns it into products:

- [`src/cleaning.py`](src/cleaning.py) + [`src/features.py`](src/features.py) — the same
  cleaning & feature pipeline the notebooks use, importable.
- [`src/train.py`](src/train.py) — trains the selected model and writes
  `reports/final_model_metrics.json`.
- [`src/regional_equity.py`](src/regional_equity.py) — regenerates the equity stretch-goal
  memo and figure from the saved final model.
- [`src/make_presentation.py`](src/make_presentation.py) — rebuilds the 5-slide deck from
  `figures/*.png` and the saved model card, so the slides can never quote a stale number.

## 🚀 Quick start

```bash
# 1. Get the code
git clone <your-repo-url>
cd ethiopian-crop-yield-revenue-predictor

# 2. Create the environment (Anaconda / any Python 3.10+)
conda create -n crop-yield-revenue python=3.10 -y
conda activate crop-yield-revenue
pip install -r requirements.txt

# 3. Launch JupyterLab
jupyter lab
```

## 📖 How to use this project

**If you want to see the result** → run the demo straight away (Step 1).

**If you want to reproduce the whole workflow** → run every notebook top to bottom in
order (Steps 2–5). Each has its own testing checkpoint in
[`NOTEBOOK_GUIDE/Documenation.md`](NOTEBOOK_GUIDE/Documenation.md).

**If you just want to understand it before touching code** → read
[`NOTEBOOK_GUIDE/Documenation.md`](NOTEBOOK_GUIDE/Documenation.md), a plain-language walkthrough
of every notebook cell, every function in `src/`, and every section of the app.

### 1. Run the demo (Deliverable E)

```bash
streamlit run app/app.py
```

Enter region, crop, year, planting month and plot details — the **growing-season weather and
the market price are looked up automatically, never typed by hand**. The app returns the
predicted yield, an estimated revenue, historical context, and three interactive charts.
*Run from a terminal with the environment active, not from inside Jupyter.*

### 2. Notebook 01 — Cleaning & integration (Deliverable A)

Run top to bottom. It produces `data/processed/master_train.csv`, `master_test.csv`,
`data_dictionary_master.csv` and `cleaning_stats.json`, plus the app assets
(`app/assets/weather_clean.csv`, `price_clean.csv`, `crop_optima.json`). The A7 cell should
print **all [PASS]** integrity checks.

### 3. Notebooks 02 & 03 — Analysis & visualizations (Deliverables B, C)

Generate the analysis report and the first 9 figures into `figures/` with
`figures/figure_captions.md` filled in.

### 4. Notebook 04 — Modeling & evaluation (Deliverable D)

Runs and screens all 108 combinations, tunes the winner, saves
`models/final_model.joblib`, writes the 3,750-row
`submission/zedingil_zelalem_submission.csv`, and adds figures 10–12 (model comparison,
residuals, permutation importance).

### 5. Export the written reports

```bash
jupyter nbconvert --to html notebooks/01_cleaning_and_integration.ipynb --output-dir reports --output A_cleaning_and_integration
jupyter nbconvert --to html notebooks/02_analysis_report.ipynb         --output-dir reports --output B_analysis_report
jupyter nbconvert --to html notebooks/04_modeling_and_evaluation.ipynb --output-dir reports --output D_model_evaluation
```

### 6. Rebuild the extras

```bash
python src/regional_equity.py     # regional-equity stretch memo + figure
python src/make_presentation.py   # the 5-slide PDF deck in presentation/
```

## 📁 Project structure

```
ethiopian-crop-yield-revenue-predictor/
├── app/                 Streamlit demo (app.py + cleaned-data assets + theme)
├── data/
│   ├── raw/              the 5 original CSVs, never edited
│   └── processed/        master_train.csv, master_test.csv, data_dictionary_master.csv
├── figures/              the 12 required PNGs + figure_captions.md
├── models/               final_model.joblib
├── notebooks/            01_cleaning → 02_analysis → 03_visualizations → 04_modeling
├── presentation/         zedingil_zelalem_slides.pdf
├── reports/              exported A/B/D reports + the regional-equity stretch memo
├── src/                  cleaning.py, features.py, train.py, regional_equity.py
├── submission/           zedingil_zelalem_submission.csv  (the graded prediction file)
├── NOTEBOOK_GUIDE/Documenation.md   plain-language guide to every notebook cell, function, and app section
└── requirements.txt      pinned environment (verified on Python 3.11.9, Windows)
```

## 🗂️ Where each deliverable lives

| Deliverable | Location |
|:---|:---|
| **A** — Pipeline | `notebooks/01_…`, `reports/A_cleaning_and_integration.html`, `data/processed/` |
| **B** — Analysis | `notebooks/02_…`, `reports/B_analysis_report.html` |
| **C** — Visualizations | `figures/`, `notebooks/03_…` |
| **D** — Modeling | `notebooks/04_…`, `reports/D_model_evaluation.html`, `models/` |
| **E** — Demo | `app/` → `streamlit run app/app.py` |
| **F** — Slides | `presentation/zedingil_zelalem_slides.pdf` |
| **G** — Structure | project root |
| **Stretch** — Equity | `reports/stretch_regional_equity.md` + `.png` |
| **Score** | `submission/zedingil_zelalem_submission.csv` |

## 🖼️ Figures at a glance

The 12 evaluation figures live in `figures/`:

| Figure | Shows |
|:---|:---|
| `fig01` | Missingness across the three raw tables (fertilizer, soil quality and labour are the most affected) |
| `fig02` | Fertilizer & price distributions before vs after cleaning |
| `fig03` | Yield distribution — right-skewed, mean 2.76 t/ha |
| `fig04` | Mean yield by region × crop heatmap — Somali sits far below the pack |
| `fig05` | Correlation heatmap — temperature gap −0.47 is the strongest correlate |
| `fig06` | Climate by region and growing-season window |
| `fig07` | Yield vs season temperature, per crop sweet spots |
| `fig08` | Price trends 2021→2024 — teff consistently most expensive |
| `fig09` | Revenue per hectare by crop & region |
| `fig10` | Validation RMSE of every model family vs the mean-predictor baseline |
| `fig11` | Predicted vs actual + residuals — centred on zero (mean +0.024 t/ha) |
| `fig12` | Permutation importance — 5 of the top 14 features are weather-derived |

## 🧪 Method hygiene (rule G5)

- The final model uses several **weather-derived features** (Rule 5): the growing-season window
  join supplies 19 model features, including `season_avg_temp_c` and the crop-temperature-gap
  features.
- **Nothing is fitted on the test file** (Rule 6): cleaning stats, crop optima, encoders, scalers
  and hyperparameter search all learn from the train file only; the test file gets the
  already-fitted constants.
- **Price is never a model feature** (G5): `market_prices.csv` is joined only for revenue
  analysis and the demo's revenue estimate, in a separate code path from feature-building.

## 📚 Documentation

- **[`NOTEBOOK_GUIDE/Documenation.md`](NOTEBOOK_GUIDE/Documenation.md)** — the plain-language
  walkthrough of *every* notebook cell, function, and app section. Start here.
- **[`reports/stretch_regional_equity.md`](reports/stretch_regional_equity.md)** — the
  regional-equity stretch-goal writeup.
- **[`figures/figure_captions.md`](figures/figure_captions.md)** — one-paragraph takeaways for
  all 12 figures.

---

<div align="center">

## 👤 About the Author

<img src="https://img.shields.io/badge/Solo-Submission-2E7D32?style=flat-square" alt="solo submission">

</div>

**Hi, I'm Zedingil Zelalem** 👋 — the author, builder and owner of this entire project,
from the first line of cleaning code to the final slide. I built it end-to-end solo for the
**Qiyas AI Hackathon 2026**: data engineering, analysis, visualization, modeling, evaluation,
the live demo and the presentation. If the model mispredicts a teff plot, that's on me too. 😄

I enjoy turning messy, real-world data (three tables, two missing-value sentinels and a price
unit mix-up — you've read the story) into something people can actually *use* to make
a decision. I'm always open to questions, feedback, or collaboration — reach out any time:

<br>

<div align="center">

| 🔗 Connection | Link |
|:---:|:---|
| 🌐 **Portfolio** | **[zedingil.vercel.app](https://zedingil.vercel.app)** |
| 🏢 **Ethio-Bacos Tech Hub** | **[ethio-bacos-tech-hub.vercel.app](https://ethio-bacos-tech-hub.vercel.app)** |

<a href="https://zedingil.vercel.app">
  <img src="https://img.shields.io/badge/Portfolio-zedingil.vercel.app-0A66C2?style=for-the-badge" alt="Portfolio">
</a>
&nbsp;
<a href="https://ethio-bacos-tech-hub.vercel.app">
  <img src="https://img.shields.io/badge/Website-ethio--bacos--tech--hub.vercel.app-2E7D32?style=for-the-badge" alt="Ethio-Bacos Tech Hub">
</a>

</div>

---

<div align="center">

*Built with 🌾, cleaned data, and 108 model combinations — by Zedingil Zelalem.*

</div>
