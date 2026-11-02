"""Charts with Russian labels for reports/A3_slides_ru.pptx.

Reads the saved A3 results (results/a3_04 … a3_07), writes PNGs to reports/a3_slides/img/.
Run from the repo root: python reports/a3_slides/charts.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(exist_ok=True)

# palette of the midterm deck
NAVY, RED, BLUE, GREEN, AMBER, GRAY, CARD, TEXT = (
    "#1F2A44", "#C53030", "#2B6CB0", "#2F855A", "#B7791F", "#718096", "#EDF2F7", "#1A202C")
LIGHT = "#A0AEC0"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "text.color": TEXT,
    "axes.edgecolor": "#CBD5E0", "axes.labelcolor": TEXT, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#4A5568", "ytick.color": "#4A5568", "axes.grid": False,
})


def ru(x, d=3):
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def ci(s):
    lo, hi = [float(v) for v in s.strip("[]").split(",")]
    return lo, hi


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Waterfall: where the gain comes from -------------------------------------------------------
allm = pd.read_csv(RES / "a3_07/all_models_test.csv").set_index("model")
base = allm.loc["Logistic regression · midterm baseline", "test PR-AUC"]
cat = allm.loc["CatBoost · tuned", "test PR-AUC"]
fin = allm.loc["Final model", "test PR-AUC"]
ens = allm.loc["Final model + LLM ensemble", "test PR-AUC"]
steps = [("Мидтерм\nлог. регрессия", 0, base, NAVY, True),
         ("Сильнее\nмодель\n(CatBoost)", base, cat - base, BLUE, False),
         ("Новая\nинформация\n(текст, история,\nцена)", cat, fin - cat, BLUE, False),
         ("Финальная\nмодель", 0, fin, RED, True),
         ("+ LLM\nв ансамбле", fin, ens - fin, LIGHT, False)]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
for i, (lab, bottom, h, c, total) in enumerate(steps):
    ax.bar(i, h, bottom=bottom, color=c, width=0.62)
    top = bottom + h
    txt = ru(top) if total else "+" + ru(h)
    ax.text(i, top + 0.006, txt, ha="center", va="bottom", fontsize=12, fontweight="bold",
            color=RED if total and c == RED else TEXT)
    if i < len(steps) - 1:
        ax.plot([i + 0.31, i + 0.69], [top, top], color=LIGHT, lw=1, ls=":")
ax.axhline(allm.loc["No model (failure rate)", "test PR-AUC"], color=GRAY, lw=1, ls="--")
ax.text(4.45, 0.072, "без модели 0,066", ha="right", va="bottom", fontsize=9, color=GRAY)
ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=9.5)
ax.set_ylim(0, 0.42)
ax.set_ylabel("PR-AUC на test")
ax.yaxis.set_major_formatter(lambda v, _: ru(v, 2))
save(fig, "waterfall.png")

# 2. Four models, default vs tuned -------------------------------------------------------------
m4 = pd.read_csv(RES / "a3_04/metrics_test.csv", index_col=0)
fams = [("Logistic regression", "Логистическая регрессия"), ("Random forest", "Случайный лес"),
        ("LightGBM", "LightGBM"), ("CatBoost", "CatBoost")]
fig, ax = plt.subplots(figsize=(6.6, 4.6))
y = np.arange(len(fams))[::-1]
for yi, (en, rus) in zip(y, fams):
    d, t = m4.loc[f"{en} · default", "PR-AUC"], m4.loc[f"{en} · tuned", "PR-AUC"]
    ax.barh(yi + 0.19, d, height=0.36, color=LIGHT)
    ax.barh(yi - 0.19, t, height=0.36, color=RED if en == "CatBoost" else BLUE)
    ax.text(d + 0.003, yi + 0.19, ru(d), va="center", fontsize=10, color=GRAY)
    ax.text(t + 0.003, yi - 0.19, ru(t), va="center", fontsize=10, fontweight="bold")
ax.axvline(base, color=NAVY, lw=1, ls="--")
ax.text(base + 0.002, 3.5, "мидтерм 0,262", ha="left", fontsize=9, color=NAVY)
ax.set_ylim(-0.6, 3.75)
ax.set_yticks(y, [r for _, r in fams])
ax.set_xlim(0.2, 0.335)
ax.set_xlabel("PR-AUC на test (20 признаков мидтерма)")
ax.xaxis.set_major_formatter(lambda v, _: ru(v, 2))
ax.legend(handles=[Patch(color=LIGHT, label="настройки по умолчанию"), Patch(color=BLUE, label="после подбора")],
          loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
save(fig, "models.png")

# 3. Optuna convergence ------------------------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.ERROR)
store = f"sqlite:///{RES / 'a3_04/optuna.db'}"
fig, ax = plt.subplots(figsize=(6.0, 4.3))
for name, rus, c in [("catboost", "CatBoost", RED), ("lightgbm", "LightGBM", BLUE), ("random_forest", "Случайный лес", GRAY)]:
    st = optuna.load_study(study_name=name, storage=store)
    v = np.array([t.value for t in st.trials if t.value is not None])
    best = np.maximum.accumulate(v)
    ax.plot(np.arange(1, len(v) + 1), best, color=c, lw=2, label=rus)
    ax.scatter(np.arange(1, len(v) + 1), v, color=c, s=8, alpha=0.35)
ax.set_xlabel("номер попытки Optuna")
ax.set_ylabel("лучший PR-AUC на validation")
ax.set_ylim(0.29, 0.355)
ax.yaxis.set_major_formatter(lambda v, _: ru(v, 2))
ax.legend(frameon=False, fontsize=9, loc="lower right")
save(fig, "optuna.png")

# 4. Text: alone (unseen names) and as an addition to the table --------------------------------
ta = pd.read_csv(RES / "a3_05/text_alone.csv").set_index("model")
tt = pd.read_csv(RES / "a3_05/text_in_tabular.csv").set_index("model")
rows = [("TF-IDF (без переноса)", "TF-IDF + logistic regression (no transfer)", GRAY),
        ("rubert-tiny2, заморожен", "frozen rubert-tiny2 + linear probe (transfer)", BLUE),
        ("multilingual-e5-small, заморожен", "frozen multilingual-e5-small + linear probe (transfer)", BLUE),
        ("rubert-base-cased, заморожен", "frozen rubert-base-cased + linear probe (transfer)", BLUE),
        ("rubert-tiny2, дообучен", "fine-tuned rubert-tiny2 (transfer + fine-tuning)", RED)]
fig, ax = plt.subplots(1, 2, figsize=(11.6, 3.5), gridspec_kw={"wspace": 0.85})
y = np.arange(len(rows))[::-1]
for yi, (lab, key, c) in zip(y, rows):
    v = ta.loc[key, "test PR-AUC, unseen texts"]
    ax[0].barh(yi, v, color=c, height=0.6)
    ax[0].text(v + 0.002, yi, ru(v), va="center", fontsize=10, fontweight="bold" if c == RED else None)
ax[0].axvline(0.0967, color=GRAY, lw=1, ls="--")
ax[0].text(0.099, 4.5, "доля срывов 0,097", fontsize=9, color=GRAY)
ax[0].set_ylim(-0.5, 4.85)
ax[0].set_yticks(y, [r[0] for r in rows])
ax[0].set_xlim(0.08, 0.245)
ax[0].set_title("Только текст: названия, которых не было в train")
ax[0].set_xlabel("PR-AUC на test")
ax[0].xaxis.set_major_formatter(lambda v, _: ru(v, 2))
add = [("32 компоненты rubert-tiny2", "+ 32 components of frozen rubert-tiny2", RED),
       ("32 компоненты e5-small", "+ 32 components of frozen multilingual-e5-small", BLUE),
       ("32 компоненты rubert-base", "+ 32 components of frozen rubert-base-cased", BLUE),
       ("оценка дообученной модели", "+ fine-tuned text score (stacking)", GRAY),
       ("смесь двух моделей", "blend: tabular + fine-tuned text (weight 0.30 on validation)", GRAY)]
y2 = np.arange(len(add))[::-1]
for yi, (lab, key, c) in zip(y2, add):
    s = tt.loc[key, "Δ test PR-AUC vs 20 features"]
    v = float(s.split()[0])
    lo, hi = ci(s[s.index("["):])
    ax[1].barh(yi, v, color=c, height=0.6, xerr=[[v - lo], [hi - v]], ecolor="#4A5568", capsize=3)
    ax[1].text(hi + 0.002, yi, "+" + ru(v), va="center", fontsize=10, fontweight="bold" if c == RED else None)
ax[1].set_yticks(y2, [a[0] for a in add])
ax[1].set_xlim(0, 0.062)
ax[1].set_title("Текст как добавка к 20 признакам (LightGBM)")
ax[1].set_xlabel("прирост PR-AUC на test, 95 % интервал")
ax[1].xaxis.set_major_formatter(lambda v, _: ru(v, 2))
save(fig, "text.png")

# 5. LLM vs boosting ---------------------------------------------------------------------------
lv = pd.read_csv(RES / "a3_06/llm_vs_boosting.csv", index_col=0)
bars = [("Qwen2.5-1.5B\nQLoRA, 40 тыс.", lv.loc["Qwen2.5-1.5B + QLoRA (40 k rows)", "test PR-AUC"], RED),
        ("LightGBM,\nте же 40 тыс.", lv.loc["LightGBM, same 40 k rows, same information", "test PR-AUC"], BLUE),
        ("LightGBM,\nвсе 266 тыс.", lv.loc["LightGBM, full train (266 k rows)", "test PR-AUC"], BLUE),
        ("LightGBM + LLM\nансамбль", allm.loc["simple LightGBM + LLM ensemble", "test PR-AUC"], NAVY)]
fig, ax = plt.subplots(figsize=(6.6, 3.7))
for i, (lab, v, c) in enumerate(bars):
    ax.bar(i, v, color=c, width=0.6)
    ax.text(i, v + 0.004, ru(v), ha="center", fontsize=12, fontweight="bold")
ax.hlines(cat, -0.5, 3.35, color=GRAY, lw=1, ls="--")
ax.text(3.4, cat, "лучшая на\n20 признаках\n0,311", ha="left", va="center", fontsize=9, color=GRAY)
ax.set_xlim(-0.5, 4.2)
ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=9.5)
ax.set_ylim(0.25, 0.375)
ax.set_ylabel("PR-AUC на test")
ax.yaxis.set_major_formatter(lambda v, _: ru(v, 2))
save(fig, "llm.png")

# 6. Ablations ---------------------------------------------------------------------------------
ab = pd.read_csv(RES / "a3_07/ablation.csv", index_col=0)
arows = [("без текста лотов", "− text components"), ("без истории «заказчик–поставщик»", "− customer–supplier history"),
         ("без цены к рынку", "− price vs market"), ("без истории и цены", "− history and price"),
         ("только 20 признаков мидтерма", "only the 20 midterm features")]
fig, ax = plt.subplots(figsize=(5.6, 3.9))
y = np.arange(len(arows))[::-1]
for yi, (lab, key) in zip(y, arows):
    s = ab.loc[key, "change vs final"]
    v = float(s.split()[0]); lo, hi = ci(s[s.index("["):])
    sig = hi < 0
    ax.barh(yi, v, color=RED if sig else LIGHT, height=0.6, xerr=[[v - lo], [hi - v]], ecolor="#4A5568", capsize=3)
    ax.text(lo - 0.002 if v < 0 else hi + 0.002, yi, ru(v), va="center", ha="right" if v < 0 else "left",
            fontsize=10, fontweight="bold" if sig else None)
ax.axvline(0, color=TEXT, lw=0.8)
ax.set_yticks(y, [a[0] for a in arows])
ax.set_xlim(-0.062, 0.02)
ax.set_xlabel("изменение PR-AUC на test, 95 % интервал")
ax.xaxis.set_major_formatter(lambda v, _: ru(v, 2))
save(fig, "ablation.png")

# 7. SHAP (mean |SHAP| of the final model on 5 000 test procurements, notebook 07 §5) ----------
shap = [("Сумма закупки", 0.532), ("Способ закупки", 0.459), ("Текст лотов (32 компоненты)", 0.302),
        ("Доля побед победителя в прошлых торгах", 0.213), ("Опыт организатора", 0.140), ("Заявок на лот", 0.122),
        ("Регион", 0.090), ("Предмет: товар, работа, услуга", 0.078), ("Лучшая цена к плановой", 0.071),
        ("Цена за единицу к рынку", 0.045)]
fig, ax = plt.subplots(figsize=(5.6, 3.9))
y = np.arange(len(shap))[::-1]
new = {"Текст лотов (32 компоненты)", "Доля побед победителя в прошлых торгах", "Цена за единицу к рынку"}
for yi, (lab, v) in zip(y, shap):
    ax.barh(yi, v, color=RED if lab in new else BLUE, height=0.65)
    ax.text(v + 0.006, yi, ru(v, 2), va="center", fontsize=9.5)
ax.set_yticks(y, [s[0] for s in shap], fontsize=9.5)
ax.set_xlim(0, 0.62)
ax.set_xlabel("средний |SHAP|, лог-шансы срыва")
ax.legend(handles=[Patch(color=BLUE, label="признаки мидтерма"), Patch(color=RED, label="новые в A3")],
          frameon=False, fontsize=9, loc="lower right")
save(fig, "shap.png")

# 8. Leakage case ------------------------------------------------------------------------------
lk = pd.read_csv(RES / "a3_07/leakage_winner.csv", index_col=0)
lrows = [("победитель = поставщик\nпервого договора", "winner = supplier of the first contract", RED),
         ("победитель = мин. БИН\nсреди договоров (ресёрч v2)", "winner = min(BIN) among contracts (research v2)", RED),
         ("победитель из объявления\nили по мин. цене (без утечки)", "winner from announcement / lowest bid (leak-free)", BLUE)]
fig, ax = plt.subplots(figsize=(5.8, 4.0))
y = np.arange(len(lrows))[::-1]
for yi, (lab, key, c) in zip(y, lrows):
    s = lk.loc[key, "gain over 20 features"]
    v = float(s.split()[0]); lo, hi = ci(s[s.index("["):])
    ax.barh(yi, v, color=c, height=0.55, xerr=[[v - lo], [hi - v]], ecolor="#4A5568", capsize=3)
    ax.text(hi + 0.002 if v > 0 else lo - 0.002, yi, ("+" if v > 0 else "") + ru(v), va="center",
            ha="left" if v > 0 else "right", fontsize=11, fontweight="bold")
ax.axvline(0, color=TEXT, lw=0.8)
ax.set_yticks(y, [l[0] for l in lrows], fontsize=9.5)
ax.set_xlim(-0.02, 0.058)
ax.set_xlabel("прирост PR-AUC к 20 признакам, 95 % интервал")
ax.xaxis.set_major_formatter(lambda v, _: ru(v, 2))
save(fig, "leak.png")

# 9. Segments: lift baseline -> final ----------------------------------------------------------
sg = pd.read_csv(RES / "a3_07/segments.csv").set_index("segment")
srows = [("Из одного источника после несост.", "trade_method = Single source after failed tender"),
         ("Запрос ценовых предложений", "trade_method = Request for quotations"),
         ("Открытый конкурс", "trade_method = Open tender"),
         ("Конкурс: школьное питание", "trade_method = Tender: school food (goods)"),
         ("Конкурс с балльной системой", "trade_method = Tender with scoring system"),
         ("Q1 — самые дешёвые", "value_quartile = Q1 (cheapest)"),
         ("Q2", "value_quartile = Q2"), ("Q3", "value_quartile = Q3"),
         ("Q4 — самые дорогие", "value_quartile = Q4 (largest)")]
fig, ax = plt.subplots(figsize=(6.4, 4.7))
y = np.arange(len(srows))[::-1].astype(float)
y[5:] -= 0.6
for yi, (lab, key) in zip(y, srows):
    b, f = sg.loc[key, "lift, baseline"], sg.loc[key, "lift, final"]
    ax.plot([b, f], [yi, yi], color=LIGHT, lw=2, zorder=1)
    ax.scatter(b, yi, color=GRAY, s=36, zorder=2)
    ax.scatter(f, yi, color=RED if f > b else NAVY, s=46, zorder=3)
    ax.text(max(b, f) + 0.18, yi, f"{ru(b, 1)} → {ru(f, 1)}", va="center", fontsize=9.5)
ax.axvline(1, color=GRAY, lw=0.8, ls=":")
ax.set_yticks(y, [s[0] for s in srows], fontsize=9.5)
ax.set_xlim(0.5, 9.6)
ax.set_xlabel("подъём = PR-AUC / доля срывов в сегменте (1 = случайно)")
ax.scatter([], [], color=GRAY, s=36, label="мидтерм"); ax.scatter([], [], color=RED, s=46, label="финальная модель")
ax.legend(frameon=False, fontsize=9, loc="lower right")
save(fig, "segments.png")

print("written:", sorted(p.name for p in OUT.glob("*.png")))
