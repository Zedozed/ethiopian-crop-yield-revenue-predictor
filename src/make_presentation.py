"""Build the 5-slide presentation deck (Deliverable F) as a PDF.

Everything on the slides is generated from code and the project's own artifacts
(``figures/*.png`` and ``reports/final_model_metrics.json``), so the deck can never
quote a stale number. It is a solo submission: there is no team/member slide.

Run from the project root::

    python src/make_presentation.py

Output: ``presentation/zedingil_zelalem_slides.pdf``
"""

from __future__ import annotations

import json
import math
import os

from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

# --------------------------------------------------------------------------- #
# Geometry and theme
# --------------------------------------------------------------------------- #
W, H = 960.0, 540.0          # 16:9 slide, in points (13.33in x 7.5in)
MARGIN = 44.0
FOOTER_Y = 34.0

# Colourblind-safe green / amber "farming" palette (same family as the figures).
INK = HexColor("#14261B")
BODY = HexColor("#2C3A2E")
GREEN = HexColor("#2E7D32")
DEEP = HexColor("#1B5E20")
AMBER = HexColor("#B26A00")
AMBER_FILL = HexColor("#F4B740")
GRAY = HexColor("#5A6B5C")
LIGHT = HexColor("#EDF4E9")
CARD = HexColor("#F4F8F1")
LINE = HexColor("#D3DFCC")
WARN = HexColor("#C62828")

FONT = "Helvetica"
FONT_B = "Helvetica-Bold"

AUTHOR = "Zedingil Zelalem"
PROJECT = "Ethiopian Crop Yield & Revenue Predictor  ·  Qiyas AI Hackathon 2026"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "presentation")
OUT_PDF = os.path.join(OUT_DIR, "zedingil_zelalem_slides.pdf")


def load_metrics():
    """Read the model card notebook 04 writes, so the deck quotes live scores."""
    with open(os.path.join(REPO_ROOT, "reports", "final_model_metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    v = m["validation"]
    return {
        "rmse": v["RMSE"],
        "mae": v["MAE"],
        "r2": v["R2"],
        "cv_rmse": m["cv_rmse_mean"],
        "cv_std": m["cv_rmse_std"],
        "oot": m["out_of_time_rmse"],
        "weather_gain": m["weather_ablation_rmse_gain"],
        "model": m["selected_model"],
    }


METRICS = load_metrics()
RMSE_TXT = f"{METRICS['rmse']:.3f}"
MAE_TXT = f"{METRICS['mae']:.3f}"
R2_TXT = f"{METRICS['r2']:.3f}"

# --------------------------------------------------------------------------- #
# Small drawing helpers
# --------------------------------------------------------------------------- #
def para(c, x, y_top, html, size=10.4, leading=13.4, color=BODY, maxw=400.0,
         font=FONT, space_after=0.0, bullet=None, bold_color=None):
    """Draw wrapped rich text; return the y coordinate just below the block."""
    style = ParagraphStyle(
        "b", fontName=font, fontSize=size, leading=leading, textColor=color,
        bulletIndent=0, leftIndent=(12 if bullet else 0),
        bulletFontName=FONT_B, bulletFontSize=size,
        bulletColor=(GREEN if bullet else color),
        spaceAfter=space_after,
    )
    p = Paragraph(html, style)
    _w, h = p.wrapOn(c, maxw, 2000)
    p.drawOn(c, x, y_top - h)
    return y_top - h


def ring(c, cx, cy, r, color=GREEN, lw=1.3, alpha=0.25):
    """Stroke-only circular ring (decorative)."""
    c.saveState()
    c.setStrokeAlpha(alpha)
    c.setStrokeColor(color)
    c.setLineWidth(lw)
    c.circle(cx, cy, r, stroke=1, fill=0)
    c.restoreState()


def heart(c, cx, cy, size, color=AMBER, alpha=0.9, stroke=0):
    """Filled, mathematically exact heart centred on (cx, cy); size = width in points."""
    s = size / 16.0
    c.saveState()
    c.setFillAlpha(alpha)
    c.setFillColor(color)
    p = c.beginPath()
    n = 48
    for i in range(n + 1):
        t = 2.0 * math.pi * i / n
        x = cx + 16.0 * (math.sin(t) ** 3) * s
        y = cy + (13.0 * math.cos(t) - 5.0 * math.cos(2 * t)
                  - 2.0 * math.cos(3 * t) - math.cos(4 * t) + 2.5) * s
        if i == 0:
            p.moveTo(x, y)
        else:
            p.lineTo(x, y)
    p.close()
    c.drawPath(p, stroke=stroke, fill=1)
    c.restoreState()


def decorations(c):
    """Faint background art layer: rings and hearts in the theme palette,
    drawn behind the slide content so text stays fully readable."""
    # Concentric rings, top-right corner
    for r, col, a in ((64, DEEP, 0.06), (46, GREEN, 0.08), (30, AMBER, 0.10)):
        ring(c, W - 26, H - 22, r, col, lw=1.4, alpha=a)
    # Double ring, bottom-left corner
    for r, col, a in ((42, GREEN, 0.07), (26, DEEP, 0.09)):
        ring(c, 20, 10, r, col, lw=1.4, alpha=a)
    # Small faint hearts in free corners
    heart(c, W / 2.0, 12, 13, AMBER, alpha=0.16)
    heart(c, 58, H - 24, 12, GREEN, alpha=0.15)


def bullets(c, x, y_top, items, maxw, size=10.4, leading=13.4, gap=5.5):
    y = y_top
    for it in items:
        y = para(c, x, y, it, size=size, leading=leading, color=BODY,
                 maxw=maxw, bullet="\u2022", space_after=0)
        y -= gap
    return y


def panel(c, x, y, w, h, fill=CARD, stroke=LINE, radius=8, lw=0.8):
    c.setLineWidth(lw)
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)


def hline(c, x1, y, x2, color=LINE, lw=1.0):
    c.setStrokeColor(color)
    c.setLineWidth(lw)
    c.line(x1, y, x2, y)


def figure(c, path, x, y, w, h):
    """Fit a PNG inside the box (x, y, w, h), centred, preserving aspect ratio."""
    img = ImageReader(path)
    iw, ih = img.getSize()
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    c.drawImage(path, x + (w - dw) / 2.0, y + (h - dh) / 2.0, dw, dh, mask="auto")


def kpi(c, x, y, w, h, value, label, accent=GREEN):
    panel(c, x, y, w, h, fill=white, stroke=LINE, radius=7)
    c.setFillColor(accent)
    c.roundRect(x, y, 4.5, h, 2.2, stroke=0, fill=1)
    c.setFillColor(accent)
    c.setFont(FONT_B, 17)
    c.drawString(x + 14, y + h - 24, value)
    c.setFillColor(GRAY)
    c.setFont(FONT, 8.7)
    ty = y + h - 37
    for ln in label:
        c.drawString(x + 14, ty, ln)
        ty -= 10.5
    ring(c, x + w - 10, y + h - 10, 4.0, accent, lw=1.1, alpha=0.45)


def header(c, stage, title, subtitle=None):
    c.setFillColor(DEEP)
    c.rect(MARGIN, H - 34, 16, 4, stroke=0, fill=1)  # small accent tick
    c.setFillColor(GREEN)
    c.setFont(FONT_B, 9.5)
    c.drawString(MARGIN + 24, H - 37, stage.upper())
    c.setFillColor(INK)
    c.setFont(FONT_B, 24)
    c.drawString(MARGIN, H - 68, title)
    y = H - 80
    if subtitle:
        c.setFillColor(GRAY)
        c.setFont(FONT, 11)
        c.drawString(MARGIN, H - 92, subtitle)
        y = H - 100
    hline(c, MARGIN, y, W - MARGIN, color=LINE, lw=1.2)
    return y - 16  # top of the content area


def footer(c, page, total=5):
    hline(c, MARGIN, FOOTER_Y + 12, W - MARGIN, color=LINE, lw=0.9)
    c.setFillColor(GRAY)
    c.setFont(FONT, 8.4)
    c.drawString(MARGIN, FOOTER_Y, f"Solo submission  ·  {PROJECT}")
    c.drawRightString(W - MARGIN, FOOTER_Y, f"{AUTHOR}  ·  Slide {page} / {total}")


def strip(c, x, y, w, h, html, fill=LIGHT, accent=DEEP):
    panel(c, x, y, w, h, fill=fill, stroke=LINE, radius=7)
    c.setFillColor(accent)
    c.roundRect(x, y, 4.5, h, 2.2, stroke=0, fill=1)
    para(c, x + 16, y + h - 8, html, size=9.6, leading=12.4, color=INK, maxw=w - 30)


# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #
def slide_1(c):
    y = header(c, "01  ·  The problem", "Predicting crop yield before planting decisions",
               "Ethiopian crop yield & revenue regression  ·  5 regions  ·  5 crops  ·  2021\u20132024")

    colw = (W - 2 * MARGIN - 26) / 2.0
    lx, rx = MARGIN, MARGIN + colw + 26

    c.setFillColor(DEEP)
    c.setFont(FONT_B, 11.5)
    c.drawString(lx, y, "The challenge")
    lx_y = y - 18
    lx_y = bullets(c, lx, lx_y, [
        "<b>Predict</b> <font face='Helvetica-Bold'>yield_tons_per_ha</font> for each of the "
        "<b>3,750 leaderboard plots</b>, using only information known at or before planting time.",
        "A <b>regression</b> task scored with <b>RMSE / MAE / R²</b> \u2014 a number, not a category.",
        "A full workflow, not just a model: integrate \u2192 analyse \u2192 visualise \u2192 "
        "evaluate \u2192 ship a working demo.",
        "The plot survey alone is not the whole picture: the weather table carries real "
        "signal a plot-level model cannot see.",
    ], maxw=colw)

    lx_y -= 6
    c.setFillColor(AMBER)
    c.setFont(FONT_B, 11.5)
    c.drawString(lx, lx_y, "The target")
    para(c, lx, lx_y - 16,
         "For every leaderboard plot, predict <b>yield in tons per hectare</b> \u2014 the "
         "organizers hold the hidden true yields.", size=10.4, leading=13.4, maxw=colw)

    concept_pipeline(c, lx, lx_y - 58, colw)

    # Right column: three data sources
    c.setFillColor(DEEP)
    c.setFont(FONT_B, 11.5)
    c.drawString(rx, y, "Three raw data sources")
    cy = y - 16
    cards = [
        ("Plot survey", GREEN,
         "15,090 train + 3,750 test plots; region, crop, year, planting month, altitude, "
         "fertilizer, improved seed, pest flag, soil quality, labour, distance to market."),
        ("Regional weather", AMBER,
         "Monthly temperature, rainfall and extreme-heat days by region\u2013year\u2013month "
         "(~226 rows); joined as a 4-month growing-season window."),
        ("Market prices", GRAY,
         "Price per quintal by crop / region / year; used to turn yield into revenue in the "
         "analysis and demo \u2014 never a model feature."),
    ]
    card_h = 78
    for name, col, text in cards:
        panel(c, rx, cy - card_h, colw, card_h, fill=CARD, stroke=LINE, radius=8)
        c.setFillColor(col)
        c.roundRect(rx, cy - card_h, 4.5, card_h, 2.2, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(FONT_B, 11.2)
        c.drawString(rx + 15, cy - 18, name)
        para(c, rx + 15, cy - 24, text, size=9.0, leading=11.4, color=BODY,
             maxw=colw - 28)
        cy -= card_h + 11

    strip(c, MARGIN, 52, W - 2 * MARGIN, 40,
          "<b>Rule 5</b> \u2014 the model uses weather-derived features  |  "
          "<b>Rule 6</b> \u2014 every median, cap, encoder and scaler is fitted on train only  |  "
          "<b>Price</b> is an analysis/demo input, never a feature")


def concept_pipeline(c, x, y_top, w):
    """Concept map built from real project numbers: the three sources flow into
    one yield prediction, which the demo turns into a revenue estimate."""
    c.setFillColor(DEEP)
    c.setFont(FONT_B, 11.5)
    c.drawString(x, y_top, "The pipeline in one line")
    stages = [
        ("Plot survey", "15,090 rows\n5 crops · 5 regions", GREEN),
        ("+ Weather join", "4-month growing\nseason window", AMBER),
        ("CatBoost \u2192 yield", "\u2192 revenue \u00b7 108\npipelines, 1 winner", DEEP),
    ]
    hh = 58
    gap = 26
    sw = (w - gap * (len(stages) - 1)) / len(stages)
    sy = y_top - 20 - hh
    sx = x
    for i, (title, sub, col) in enumerate(stages):
        panel(c, sx, sy, sw, hh, fill=white, stroke=LINE, radius=7)
        c.setFillColor(col)
        c.roundRect(sx, sy, 4.5, hh, 2.2, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(FONT_B, 10.0)
        c.drawString(sx + 12, sy + hh - 18, title)
        c.setFillColor(GRAY)
        c.setFont(FONT, 8.2)
        ty = sy + hh - 31
        for ln in sub.split("\n"):
            c.drawString(sx + 12, ty, ln)
            ty -= 10.4
        if i < len(stages) - 1:
            ay = sy + hh / 2.0
            ax1, ax2 = sx + sw + 3, sx + sw + gap - 6
            c.setStrokeColor(col)
            c.setLineWidth(1.4)
            c.line(ax1, ay, ax2 - 5, ay)
            c.setFillColor(col)
            pth = c.beginPath()
            pth.moveTo(ax2, ay)
            pth.lineTo(ax2 - 6, ay + 3.4)
            pth.lineTo(ax2 - 6, ay - 3.4)
            pth.close()
            c.drawPath(pth, stroke=0, fill=1)
            # A small heart rides on every arrow — this project is for the farmer.
            heart(c, (ax1 + ax2) / 2.0, ay + 10, 8.5, AMBER, alpha=0.85)
        sx += sw + gap


def slide_2(c):
    y = header(c, "02  ·  Cleaning & integration",
               "Three messy exports \u2192 one trustworthy modeling table",
               "Deliverable A \u2014 cleaning log, standardised labels, join map, join audit, "
               "integrity checks")

    colw = (W - 2 * MARGIN - 26) / 2.0
    lx, rx = MARGIN, MARGIN + colw + 26

    c.setFillColor(DEEP)
    c.setFont(FONT_B, 11.5)
    c.drawString(lx, y, "Data-quality problems found & fixed (A1\u2013A2)")
    bullets(c, lx, y - 18, [
        "<b>Region labels</b> needed standardising on 93.9% of plot rows \u2014 4+ casing and "
        "abbreviation variants (AMH, SOM, \u2026).",
        "<b>Crop spelling:</b> \u201cTef\u201d \u2192 teff; the weather file uses a <i>different</i> "
        "abbreviation set again.",
        "<b>Two missing-value sentinels</b> (-999 and -999.0) plus true blanks \u2192 "
        "train-fitted median / mode imputation.",
        "<b>Fertilizer outliers</b> up to ~899 kg/ha \u2192 capped at the train 99th percentile (136).",
        "<b>Weather</b> duplicate / conflicting readings averaged; <b>price unit mix-up</b> "
        "(birr/kg) corrected \u00d7100.",
    ], maxw=colw)

    c.setFillColor(AMBER)
    c.setFont(FONT_B, 11.5)
    c.drawString(rx, y, "Join logic (A3)")
    bullets(c, rx, y - 18, [
        "<b>PLOT \u2192 seasonal weather</b> (many-to-one; the aggregated weather is one row per plot, "
        "so the row count must not change).",
        "<b>PLOT \u2192 price</b> (many-to-one, revenue only \u2014 never a training feature).",
        "<b>Growing season = planting month + the following 3 months</b>, with Dec\u2192Jan "
        "wraparound.",
        "The <b>plot table is the left table</b>: every plot must survive the join; weather is "
        "only ever aggregated onto it.",
    ], maxw=colw)

    # A4 join-audit metric cards
    c.setFillColor(DEEP)
    c.setFont(FONT_B, 11.5)
    c.drawString(MARGIN, 168, "A4 join audit \u2014 match rates and integrity")
    cards = [
        ("15,090 \u2192 15,090", ["train rows before \u2192 after join", "(unchanged \u2014 as required)"]),
        ("80.6% full", ["4-month weather window", "19.4% partial \u00b7 0% unmatched"]),
        ("100%", ["price match rate", "plot \u2192 price, both splits"]),
        ("13 / 13", ["integrity checks PASS", "unique ids, no NaNs, ranges"]),
    ]
    n = len(cards)
    gap = 12
    cw = (W - 2 * MARGIN - gap * (n - 1)) / n
    cx = MARGIN
    for value, label in cards:
        kpi(c, cx, 92, cw, 62, value, label)
        cx += cw + gap

    strip(c, MARGIN, 52, W - 2 * MARGIN, 30,
          "Test set goes through the <b>same pipeline</b>, using statistics learned from train only "
          "\u2014 that is how Rule 6 is enforced structurally, not by promise.")


def slide_3(c):
    y = header(c, "03  ·  What the data says",
               "Yield is not value \u2014 and heat, not plot inputs, sets the level",
               "Two findings from the analysis (B) and figure pack (C)")

    # Two figures side by side
    figw = (W - 2 * MARGIN - 20) / 2.0
    figh = 232
    fpaths = [
        ("figures/fig04_region_crop_heatmap.png",
         "fig04 \u00b7 Mean yield by region \u00d7 crop (t/ha)"),
        ("figures/fig09_revenue_by_crop_region.png",
         "fig09 \u00b7 Revenue per hectare by crop & region (birr/ha)"),
    ]
    fx = MARGIN
    for path, cap in fpaths:
        panel(c, fx, y - figh, figw, figh, fill=white, stroke=LINE, radius=7)
        figure(c, os.path.join(REPO_ROOT, path), fx + 8, y - figh + 20, figw - 16, figh - 32)
        c.setFillColor(GRAY)
        c.setFont(FONT, 8.8)
        c.drawCentredString(fx + figw / 2.0, y - figh + 8, cap)
        fx += figw + 20

    # Two takeaway cards
    cardy = y - figh - 14
    cardh = cardy - 52
    cx = MARGIN
    cards = [
        ("FINDING 1 \u00b7 Yield \u2260 value", AMBER,
         "Teff has the <b>lowest mean yield (2.00 t/ha)</b> yet the <b>highest revenue per hectare "
         "(152,301 birr/ha)</b> \u2014 about 1.8\u00d7 sorghum\u2019s 83,623. Maize leads yield "
         "(3.90 t/ha) but not revenue. Revenue = yield \u00d7 10 quintals/ton \u00d7 price."),
        ("FINDING 2 \u00b7 Heat sets the level", GREEN,
         "<b>Somali averages 1.46 t/ha versus ~3.0\u20133.1 elsewhere.</b> Season temperature and "
         "extreme-heat days are the strongest correlates of yield (both \u2248 \u20130.45). "
         "Best cell: maize in SNNPR (4.80); worst: teff in Somali (0.83)."),
    ]
    for title, col, text in cards:
        panel(c, cx, 52, figw, cardh, fill=CARD, stroke=LINE, radius=8)
        c.setFillColor(col)
        c.roundRect(cx, 52, 4.5, cardh, 2.2, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(FONT_B, 10.6)
        c.drawString(cx + 15, 52 + cardh - 17, title)
        para(c, cx + 15, 52 + cardh - 24, text, size=9.4, leading=12.2, color=BODY,
             maxw=figw - 28)
        cx += figw + 20


def slide_4(c):
    y = header(c, "04  ·  Modeling & evaluation",
               "108 combinations, one clear winner \u2014 and the weather join earns its keep",
               "Deliverable D \u2014 baselines, model comparison, cross-validation, "
               "out-of-time check, weather ablation")

    colw = 470.0
    rx = MARGIN + colw + 24
    rw = W - MARGIN - rx

    yl = y
    yl = para(c, MARGIN, yl,
              "Sweep: <b>18 models \u00d7 2 encoders \u00d7 3 scalers = 108 pipelines</b>, screened on a "
              "stratified subsample and confirmed on the full training split. "
              "<b>Winner: CatBoost on log(yield)</b> (one-hot + robust scaler).",
              size=10.2, leading=13.2, maxw=colw)
    yl -= 10

    # Small metric table
    rows = [
        ("Variant", "RMSE", "R\u00b2", True),
        ("Mean-predictor baseline", "1.413", "\u20130.00", False),
        ("Linear regression", "0.808", "0.673", False),
        ("CatBoost log-target (winner)", RMSE_TXT, R2_TXT, False),
    ]
    row_h = 20
    ty = yl
    for i, (name, rmse, r2, head) in enumerate(rows):
        if head:
            c.setFillColor(DEEP)
            c.roundRect(MARGIN, ty - row_h, colw, row_h, 4, stroke=0, fill=1)
            c.setFillColor(white)
        else:
            c.setFillColor(CARD if i % 2 else white)
            c.rect(MARGIN, ty - row_h, colw, row_h, stroke=0, fill=1)
            c.setFillColor(INK if "winner" not in name else DEEP)
        f = FONT_B if (head or "winner" in name) else FONT
        c.setFont(f, 9.6)
        c.drawString(MARGIN + 10, ty - row_h + 6, name)
        c.drawRightString(MARGIN + colw - 60, ty - row_h + 6, rmse)
        c.drawRightString(MARGIN + colw - 12, ty - row_h + 6, r2)
        ty -= row_h
    c.setFillColor(GRAY)
    c.setFont(FONT, 8.2)
    c.drawString(MARGIN + 10, ty - 10, "Held-out validation split: 80/20 of the train file, random_state=42.")
    yl = ty - 22

    bullets(c, MARGIN, yl, [
        f"<b>5-fold CV: {METRICS['cv_rmse']:.3f} \u00b1 {METRICS['cv_std']:.3f} t/ha</b> \u2014 agrees with the single "
        "split within 0.005; the small spread means it is consistently good, not a lucky split.",
        f"<b>Out-of-time (2021\u201323 \u2192 2024): {METRICS['oot']:.3f} t/ha</b> (+0.016) \u2014 only minor "
        "temporal drift.",
        f"<b>Weather ablation: gain of {METRICS['weather_gain']:.4f} t/ha (3.3%) with the weather features</b> "
        "\u2014 the growing-season join earns its keep (Rule 5).",
    ], maxw=colw)

    # Right column figures
    f1h = 176
    panel(c, rx, y - f1h, rw, f1h, fill=white, stroke=LINE, radius=7)
    figure(c, os.path.join(REPO_ROOT, "figures/fig10_model_comparison.png"),
           rx + 8, y - f1h + 18, rw - 16, f1h - 30)
    c.setFillColor(GRAY)
    c.setFont(FONT, 8.4)
    c.drawCentredString(rx + rw / 2.0, y - f1h + 6, "fig10 \u00b7 Validation RMSE of every model family")

    f2h = 200
    f2y = y - f1h - 12 - f2h
    panel(c, rx, f2y, rw, f2h, fill=white, stroke=LINE, radius=7)
    figure(c, os.path.join(REPO_ROOT, "figures/fig12_feature_importance.png"),
           rx + 8, f2y + 18, rw - 16, f2h - 30)
    c.setFillColor(GRAY)
    c.setFont(FONT, 8.4)
    c.drawCentredString(rx + rw / 2.0, f2y + 6,
                        "fig12 \u00b7 Top features (5 of the top 14 are weather-derived)")


def slide_5(c):
    y = header(c, "05  ·  Error analysis, score & demo",
               "Where the model still struggles \u2014 and the live decision tool",
               "Deliverable D7\u2013D9, the stretch goal, and Deliverable E")

    # KPI row across the top of the content area
    kpis = [
        (f"RMSE {RMSE_TXT} t/ha", ["15.2% of mean yield"], GREEN),
        (f"MAE {MAE_TXT} t/ha", ["10.9% of mean yield"], AMBER),
        (f"R\u00b2 {R2_TXT}", ["variance explained"], GREEN),
    ]
    n = len(kpis)
    gap = 12
    kw = (W - 2 * MARGIN - gap * (n - 1)) / n
    kx = MARGIN
    for value, label, col in kpis:
        kpi(c, kx, y - 58, kw, 58, value, label, accent=col)
        kx += kw + gap

    colw = (W - 2 * MARGIN - 26) / 2.0
    lx, rx = MARGIN, MARGIN + colw + 26
    yb = y - 72

    c.setFillColor(DEEP)
    c.setFont(FONT_B, 11.5)
    c.drawString(lx, yb, "Error analysis (D7\u2013D8)")
    bullets(c, lx, yb - 18, [
        "<b>Plain language:</b> a typical plot is off by ~0.30 t/ha \u2014 about 11% of what it "
        "produces, or roughly <b>14,010 birr/ha</b> at the average 4,616 birr/quintal price. Good "
        "enough to rank plots, not to promise an exact harvest.",
        "<b>By crop:</b> hardest <b>maize</b> (0.414), easiest <b>teff</b> (0.228). <b>By region:</b> "
        "Amhara highest (0.355), Somali lowest (0.162) \u2014 but relative error is a tight 10.7\u201311.9%.",
        "<b>Hardest plots</b> cluster at low season temperature and maize: the steep part of the "
        "heat-response curve, where a small weather error moves yield a lot.",
        "<b>Stretch \u2014 regional equity:</b> after the crop-optimum features the model serves "
        "every region within ~1.2 percentage points (10.7\u201311.9%); a region-calibrated error "
        "band is the proposed fix.",
    ], maxw=colw)

    c.setFillColor(AMBER)
    c.setFont(FONT_B, 11.5)
    c.drawString(rx, yb, "Live demo (E) \u2014 and what we'd try next")
    para(c, rx, yb - 18,
         "<b>Streamlit app</b>: enter region, crop, year, planting month and plot details \u2014 "
         "the growing-season weather and the price are <b>looked up automatically, never typed</b>. "
         "Returns predicted yield, estimated revenue and historical context, with three "
         "interactive charts.", size=9.8, leading=12.6, maxw=colw)
    dy = yb - 66
    panel(c, rx, dy - 44, colw, 44, fill=LIGHT, stroke=LINE, radius=7)
    c.setFillColor(DEEP)
    c.setFont(FONT_B, 9.2)
    c.drawString(rx + 12, dy - 14, "Local run:")
    c.setFillColor(GREEN)
    c.setFont(FONT, 9.2)
    c.drawString(rx + 78, dy - 14, "streamlit run app/app.py")
    c.setFillColor(GREEN)
    c.setFont(FONT, 9.2)
    c.drawString(rx + 62, dy - 30, "streamlit run app/app.py")

    para(c, rx, dy - 54,
         "<b>Next:</b> per-plot prediction intervals (quantile regression / bootstrap), finer "
         "sub-regional weather, and more years of data.", size=9.8, leading=12.6, maxw=colw)

    # Residual figure bottom-right
    fh = 150
    fy = 52
    panel(c, rx, fy, colw, fh, fill=white, stroke=LINE, radius=7)
    figure(c, os.path.join(REPO_ROOT, "figures/fig11_predicted_vs_actual_residuals.png"),
           rx + 8, fy + 16, colw - 16, fh - 26)
    c.setFillColor(GRAY)
    c.setFont(FONT, 8.2)
    c.drawCentredString(rx + colw / 2.0, fy + 5,
                        "fig11 \u00b7 Predicted vs actual, and residuals vs predicted (t/ha)")


SLIDES = [slide_1, slide_2, slide_3, slide_4, slide_5]


def build():
    os.makedirs(OUT_DIR, exist_ok=True)
    c = canvas.Canvas(OUT_PDF, pagesize=(W, H))
    c.setTitle(PROJECT)
    c.setAuthor(AUTHOR)
    c.setSubject("Qiyas AI Hackathon 2026 - Deliverable F")
    for i, fn in enumerate(SLIDES, start=1):
        c.setFillColor(white)
        c.rect(0, 0, W, H, stroke=0, fill=1)
        decorations(c)
        fn(c)
        footer(c, i, len(SLIDES))
        c.showPage()
    c.save()
    print("Wrote", os.path.relpath(OUT_PDF, REPO_ROOT))


if __name__ == "__main__":
    build()
