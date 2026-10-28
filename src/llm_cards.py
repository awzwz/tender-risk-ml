"""
Russian text cards for the LLM (A3 criterion 5): everything the final tabular model knows about a procurement —
the 20 midterm features, the customer–supplier history and the price against the market (src/features.py) — written
as a short paragraph, followed by the lot names. The LLM therefore sees the same information as LightGBM.

    from src.llm_cards import make_card
    df["card"] = df.apply(make_card, axis=1)
"""
from __future__ import annotations

import math

import pandas as pd

from src.labels_ru import RU_METHOD, RU_REGION, RU_SUBJECT

FLAGS = {"construction": "строительство", "light_industry": "лёгкая промышленность", "disabled_org": "организация инвалидов",
         "any_design_docs": "есть проектно-сметная документация", "any_consulting": "консультационные услуги",
         "any_dumping": "действуют антидемпинговые правила", "single_organizer": "единый организатор"}
MAX_LOT_CHARS = 400


def _num(v: float, digits: int = 0) -> str:
    s = f"{v:,.{digits}f}".replace(",", " ")
    return s.replace(".", ",")


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def make_card(r: pd.Series) -> str:
    parts = [f"Способ: {RU_METHOD.get(r.trade_method, r.trade_method)}.",
             f"Предмет: {RU_SUBJECT.get(r.subject_type, r.subject_type)}.",
             f"Регион: {RU_REGION.get(r.region, r.region).rstrip('.')}.",
             f"Сумма: {_num(10 ** r.log_total_sum)} тенге, лотов: {int(r.n_lots)}."]
    if r.no_bids:
        parts.append("Заявок не было.")
    else:
        bid = f"Участников: {int(r.n_bidders)}, заявок на лот: {_num(r.bids_per_lot, 1)}"
        if _ok(r.best_bid_ratio):
            bid += f", лучшая цена к плановой: {_num(100 * r.best_bid_ratio)} %"
        parts.append(bid + ".")
    if _ok(r.bid_window_days):
        parts.append(f"Приём заявок: {_num(r.bid_window_days, 1)} дн.")
    parts.append(f"Закупок организатора с начала года: {int(r.org_prior_buys)}.")
    flags = [t for k, t in FLAGS.items() if r.get(k, 0) == 1]
    if flags:
        parts.append("Особенности: " + ", ".join(flags) + ".")
    if r.get("pair_prior_contracts", 0) > 0:
        parts.append(f"Этот заказчик раньше заключал договоры с победителем: {int(r.pair_prior_contracts)}.")
    if _ok(r.get("winner_win_rate")):
        parts.append(f"Доля побед победителя в прошлых закупках: {_num(100 * r.winner_win_rate)} %.")
    if _ok(r.get("unit_price_vs_market")):
        parts.append(f"Цена за единицу к рыночной: ×{_num(math.exp(r.unit_price_vs_market), 2)}.")
    lots = (r.lot_names or "").replace(" | ", "; ")
    if len(lots) > MAX_LOT_CHARS:
        lots = lots[:MAX_LOT_CHARS].rsplit(" ", 1)[0] + "…"
    parts.append(f"Лоты: {lots}")
    return " ".join(parts)
