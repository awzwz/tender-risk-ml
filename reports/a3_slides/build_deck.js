// Builds reports/A3_slides_ru.pptx (Assignment 3 defense, Russian) in the style of the midterm deck.
// Charts are PNGs from charts.py (Keynote does not render pptxgenjs native charts).
//
//   python reports/a3_slides/charts.py
//   NODE_PATH=<folder with pptxgenjs + jszip> node reports/a3_slides/build_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");

const HERE = __dirname;
const IMG = path.join(HERE, "img");
const OUT = path.join(HERE, "..", "A3_slides_ru.pptx");

const THEME = {
  name: "Tender Risk ML",
  headFontFace: "Georgia",
  bodyFontFace: "Arial",
  colors: {
    dk1: "1A202C", lt1: "FFFFFF", dk2: "1F2A44", lt2: "EDF2F7",
    accent1: "C53030", accent2: "2B6CB0", accent3: "2F855A", accent4: "B7791F",
    accent5: "718096", accent6: "1F2A44", hlink: "2B6CB0", folHlink: "718096",
  },
};
const MUTED = "4A5568";      // secondary text on white
const ON_DARK = "CBD5E0";    // secondary text on navy
const FOOTER = "Tender Risk ML · Assignment 3 · Обучение и дообучение моделей";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in, as the midterm deck
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Aziz Abdulin";
pres.title = "Assignment 3 — обучение и дообучение моделей";
const C = pres.SchemeColor;

const L = 0.6, R = 13.333 - 0.6, W = R - L;   // content frame
const TOP = 1.4;                               // first content row under the title

const title = { x: L, y: 0.38, w: W, h: 0.8, fontSize: 30, bold: true, align: "left", valign: "middle", margin: 0 };
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", ...title, color: C.text1 }, text: "" } },
    { text: { text: FOOTER, options: { x: L, y: 7.02, w: 8, h: 0.28, fontSize: 9, color: THEME.colors.accent5, margin: 0 } } },
  ],
  slideNumber: { x: R - 0.6, y: 7.02, w: 0.6, h: 0.28, fontSize: 9, color: THEME.colors.accent5, align: "right" },
});
pres.defineSlideMaster({ title: "COVER", background: { color: C.text2 }, objects: [] });
pres.defineSlideMaster({
  title: "CLOSING",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", ...title, color: C.background1 }, text: "" } },
    { text: { text: FOOTER, options: { x: L, y: 7.02, w: 8, h: 0.28, fontSize: 9, color: THEME.colors.accent5, margin: 0 } } },
  ],
  slideNumber: { x: R - 0.6, y: 7.02, w: 0.6, h: 0.28, fontSize: 9, color: THEME.colors.accent5, align: "right" },
});

// ---------- helpers ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function pic(slide, name, x, y, w, hMax, alt) {
  const file = path.join(IMG, name);
  const [pw, ph] = pngSize(file);
  let h = (w * ph) / pw;
  if (hMax && h > hMax) { h = hMax; w = (h * pw) / ph; }
  slide.addImage({ path: file, x, y, w, h, altText: alt, objectName: name });
  return { w, h };
}
// keep "358 789", "1 %", "≈ 280" on one line
const nb = (t) => t.replace(/(\d) (?=\d{3}\b)/g, "$1\u00A0").replace(/(\d) %/g, "$1\u00A0%").replace(/≈ /g, "≈\u00A0");
function txt(slide, runs, opts) {
  const fix = typeof runs === "string" ? nb(runs) : runs.map((r) => ({ ...r, text: nb(r.text) }));
  slide.addText(fix, { isTextBox: true, margin: 0, valign: "top", ...opts });
}
// a light (or navy) card with a bold heading and body text
function card(slide, x, y, w, h, head, body, o = {}) {
  const dark = o.dark;
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: dark ? C.text2 : C.background2 }, line: { type: "none" }, objectName: o.name || head });
  const runs = [];
  if (head) runs.push({ text: head, options: { bold: true, fontSize: o.headSize || 14, color: o.headColor || (dark ? C.background1 : C.text1), breakLine: true } });
  if (head) runs.push({ text: " ", options: { fontSize: 5, breakLine: true } });
  runs.push({ text: body, options: { fontSize: o.size || 12, color: dark ? C.background1 : C.text1 } });
  txt(slide, runs, { x: x + 0.2, y: y + 0.16, w: w - 0.4, h: h - 0.28, lineSpacingMultiple: 1.08 });
}
// numbered red circle (the deck's motif) with heading and description
function step(slide, n, x, y, w, head, body, o = {}) {
  slide.addShape(pres.shapes.OVAL, { x, y: y + 0.02, w: 0.42, h: 0.42, fill: { color: o.badge || C.accent1 }, line: { type: "none" }, objectName: `badge ${n}` });
  txt(slide, String(n), { x, y: y + 0.02, w: 0.42, h: 0.42, fontSize: o.badgeSize || 12, bold: true, color: C.background1, align: "center", valign: "middle" });
  txt(slide, [
    { text: head, options: { bold: true, fontSize: 13, color: o.color || C.text1, breakLine: true } },
    { text: body, options: { fontSize: 11.5, color: o.muted || MUTED } },
  ], { x: x + 0.6, y, w: w - 0.6, h: o.h || 0.95, lineSpacingMultiple: 1.05 });
}
function label(slide, text, x, y, w, color) {
  txt(slide, text.toUpperCase(), { x, y, w, h: 0.3, fontSize: 10.5, bold: true, color: color || C.accent1, charSpacing: 3 });
}
function caption(slide, text, x, y, w, h = 0.5) {
  txt(slide, text, { x, y, w, h, fontSize: 10.5, color: MUTED, italic: true });
}
function content(head, notes) {
  const s = pres.addSlide({ masterName: "CONTENT" });
  s.addText(head, { placeholder: "title" });
  s.addNotes(notes);
  return s;
}
const HDR = { bold: true, color: "FFFFFF", fill: { color: THEME.colors.dk2 }, fontSize: 11 };
function table(slide, rows, opts) {
  slide.addTable(rows, {
    fontSize: 11, color: THEME.colors.dk1, valign: "middle", margin: [3, 6, 3, 6],
    border: { type: "solid", pt: 0.75, color: "CBD5E0" }, autoPage: false, ...opts,
  });
}

// ---------- 1. cover ----------
{
  const s = pres.addSlide({ masterName: "COVER" });
  txt(s, "Что на самом деле улучшает прогноз срыва договоров", { x: L, y: 0.7, w: 7.4, h: 1.75, fontSize: 36, bold: true, color: C.background1, fontFace: "Georgia", valign: "bottom" });
  txt(s, "Assignment 3 · Обучение и дообучение моделей · Tender Risk ML · Азиз Абдулин", { x: L, y: 2.65, w: 7.4, h: 0.35, fontSize: 14, color: ON_DARK });
  label(s, "Задача", L, 3.55, 7.2);
  txt(s, "Та же, что в мидтерме: в момент выбора победителя предсказать, будет ли договор расторгнут или не заключён. Вопрос A3 — что даёт прирост: модель, настройка или новая информация.",
    { x: L, y: 3.9, w: 7.2, h: 1.0, fontSize: 14, color: C.background1, lineSpacingMultiple: 1.1 });
  label(s, "Главный вывод", L, 5.15, 7.2);
  txt(s, "Сильная модель и новая информация (прежде всего текст лотов) дали примерно по +0,05 PR-AUC. Подбор гиперпараметров почти ничего не добавил.",
    { x: L, y: 5.5, w: 7.2, h: 1.0, fontSize: 14, color: C.background1, lineSpacingMultiple: 1.1 });
  const stats = [
    ["0,262 → 0,364", "PR-AUC на test: мидтерм → финальная модель", C.background1],
    ["72 %", "срывов среди 1 % самых рискованных закупок (у мидтерма 50 %)", C.accent1],
    ["≈ 280 вместо 196", "найденных срывов, если проверить 1 % закупок за две недели test", C.background1],
  ];
  stats.forEach(([big, small, color], i) => {
    const y = 0.7 + i * 2.0;
    s.addShape(pres.shapes.RECTANGLE, { x: 8.55, y, w: 4.18, h: 1.8, fill: { color: C.background1, transparency: 91 }, line: { type: "none" }, objectName: `stat ${i + 1}` });
    txt(s, big, { x: 8.8, y: y + 0.2, w: 3.8, h: 0.75, fontSize: 30, bold: true, color, valign: "middle" });
    txt(s, small, { x: 8.8, y: y + 1.0, w: 3.75, h: 0.65, fontSize: 12, color: ON_DARK });
  });
  s.addNotes("Это продолжение мидтерма: задача, данные и разбиение те же. В мидтерме был baseline — логистическая регрессия с PR-AUC 0,262. В A3 я сравнил четыре модели, подобрал гиперпараметры, дообучил текстовую нейросеть и LLM. Итог: финальная модель 0,364. Если аудитор проверит 1 % закупок, он найдёт около 280 сорванных договоров вместо 196. Главный вывод: прирост дали модель и новая информация, а не подбор настроек.");
}

// ---------- 2. result and plan ----------
{
  const s = content("Прирост поровну дали сильная модель и новая информация",
    "Это главный слайд. Слева — откуда взялся прирост. Переход от линейной модели к CatBoost на тех же 20 признаках даёт +0,049. Затем тот же CatBoost получает новую информацию — текст лотов, историю «заказчик–поставщик», цену к рынку — ещё +0,053. LLM в ансамбле добавляет +0,004. Справа — как работа разбита по ноутбукам и критериям. Данные, разбиение и метрики — как в мидтерме, поэтому все цифры сравнимы.");
  const im = pic(s, "waterfall.png", L, TOP, 7.0, 4.5, "Водопад PR-AUC от мидтерма к финальной модели");
  caption(s, "PR-AUC на test (16–31 марта 2024). Данные, разбиение по времени и метрики — как в мидтерме: 358 789 закупок, срывов 6,6 %.", L, TOP + im.h + 0.15, 7.0, 0.55);
  const x = 8.0, w = R - x;
  txt(s, "Что сделано в A3", { x, y: TOP, w, h: 0.35, fontSize: 16, bold: true, color: C.text1 });
  const rows = [
    ["04", "Четыре модели, подбор, порог, калибровка", "GridSearch и Optuna · критерии 1, 2, 6"],
    ["05", "Текст лотов: перенос и дообучение", "3 замороженных кодировщика, дообученный rubert-tiny2 · критерии 3, 4"],
    ["06", "LLM с QLoRA на Kaggle", "Qwen2.5-1.5B читает карточку закупки · критерий 5"],
    ["07", "Финальная модель и разбор", "общая таблица, абляции, SHAP, утечка, сегменты · критерии 7–9"],
  ];
  rows.forEach(([n, h, b], i) => step(s, n, x, TOP + 0.55 + i * 0.93, w, h, b, { badgeSize: 10.5, h: 0.85 }));
  card(s, x, 5.75, w, 1.0, null, "Подбор гиперпараметров: от +0,001 (LightGBM) до +0,04 (случайный лес). Потолок задаёт информация, а не настройки.", { dark: true, size: 12.5 });
}

// ---------- 3. model comparison ----------
{
  const s = content("Бустинг сильнее регрессии, подбор добавляет мало",
    "Критерии 1 и 2. Четыре модели на тех же 20 признаках, что в мидтерме: логистическая регрессия, случайный лес, LightGBM и CatBoost. Серые столбцы — настройки по умолчанию, цветные — после подбора. Подбор шёл только на validation, test открывался один раз. Лучшая модель — CatBoost после подбора: 0,311. Интересно, что LightGBM уже по умолчанию даёт 0,303, и подбор ему почти ничего не добавил: ранняя остановка оставляет всего около 14 деревьев.");
  pic(s, "models.png", L, TOP, 6.9, 4.9, "PR-AUC четырёх моделей до и после подбора");
  const x = 7.9, w = R - x;
  card(s, x, TOP, w, 1.55, "Как сравнивали", "4 модели на тех же 20 признаках, что в мидтерме. Подбор только на validation (1–15 марта), test открывается один раз.");
  card(s, x, TOP + 1.75, w, 1.75, "Как подбирали", "Логистическая регрессия — GridSearch, 42 сочетания (C, штраф, веса классов). Лес, LightGBM и CatBoost — Optuna (TPE), 30–50 попыток.");
  card(s, x, TOP + 3.7, w, 1.75, "Вывод", "Лучшая — CatBoost после подбора: 0,311 против 0,262 (+0,049, интервал [+0,039; +0,059]). LightGBM по умолчанию уже 0,303: подбор дал +0,001.", { dark: true });
}

// ---------- 4. tuning, threshold, calibration ----------
{
  const s = content("Подбор выходит на плато, калибровка чинит вероятности",
    "Критерии 2 и 6: до и после. Слева — как росло качество на validation по ходу подбора Optuna. CatBoost вышел на максимум к 24-й попытке, дальше почти ничего. Справа сверху — порог: стандартный 0,5 почти никого не помечает, поэтому порог подбирается на validation — по максимуму F1 или под бюджет проверок 5 %. Снизу — калибровка: модель мидтерма с весами классов завышала вероятности в шесть раз; масштабирование Платта приводит их к реальной доле срывов, а ранжирование не меняет.");
  const im = pic(s, "optuna.png", L, TOP, 6.3, 4.6, "Сходимость Optuna по попыткам");
  caption(s, "Точки — отдельные попытки, линия — лучший результат к этой попытке. CatBoost достиг максимума на 24-й попытке из 30, LightGBM — на 44-й из 50.", L, TOP + im.h + 0.12, 6.2, 0.6);
  const x = 7.3, w = R - x;
  txt(s, "Порог решения (CatBoost после подбора)", { x, y: TOP, w, h: 0.32, fontSize: 14, bold: true, color: C.text1 });
  const c = (t, o = {}) => ({ text: t, options: o });
  table(s, [
    [c("Порог", HDR), c("Помечено", { ...HDR, align: "right" }), c("Точность", { ...HDR, align: "right" }), c("Полнота", { ...HDR, align: "right" })],
    [c("0,5 по умолчанию"), c("0,6 %", { align: "right" }), c("0,60", { align: "right" }), c("0,05", { align: "right" })],
    [c("макс. F1 на validation (0,188)", { bold: true }), c("8,0 %", { align: "right", bold: true }), c("0,34", { align: "right", bold: true }), c("0,41", { align: "right", bold: true })],
    [c("бюджет 5 % (0,249)"), c("4,8 %", { align: "right" }), c("0,41", { align: "right" }), c("0,30", { align: "right" })],
  ], { x, y: TOP + 0.45, w, colW: [2.55, 0.96, 0.96, 0.96], rowH: 0.36 });
  txt(s, "Калибровка: масштабирование Платта", { x, y: TOP + 2.2, w, h: 0.32, fontSize: 14, bold: true, color: C.text1 });
  const half = (w - 0.2) / 2;
  [["0,415 → 0,069", "средняя оценка модели мидтерма при доле срывов 0,066"],
   ["0,200 → 0,055", "Brier score мидтерма; ранжирование и PR-AUC не меняются"]].forEach(([big, small], i) => {
    const xx = x + i * (half + 0.2);
    s.addShape(pres.shapes.RECTANGLE, { x: xx, y: TOP + 2.65, w: half, h: 1.55, fill: { color: C.background2 }, line: { type: "none" }, objectName: `calib ${i + 1}` });
    txt(s, big, { x: xx + 0.18, y: TOP + 2.8, w: half - 0.3, h: 0.5, fontSize: 22, bold: true, color: C.accent2 });
    txt(s, small, { x: xx + 0.18, y: TOP + 3.35, w: half - 0.3, h: 0.8, fontSize: 11.5, color: MUTED });
  });
  txt(s, "Isotonic-калибровка создаёт ничьи в оценках и снижает PR-AUC (0,311 → 0,303), поэтому выбран Платт.", { x, y: TOP + 4.4, w, h: 0.6, fontSize: 11.5, color: C.text1 });
}

// ---------- 5. text: transfer learning vs fine-tuning ----------
{
  const s = content("Текст: дообучение лучше всех читает незнакомые названия",
    "Критерии 3 и 4. Слева — модели, которые видят только названия лотов, на закупках с названиями, которых не было в train. Это самая честная проверка понимания текста. Замороженные кодировщики с линейной моделью сверху — это transfer learning — слабее даже TF-IDF: названия короткие и шаблонные. Дообучение rubert-tiny2 целиком даёт лучший результат — 0,228. Справа — текст как добавка к 20 признакам. Здесь лучше всего работают 32 компоненты замороженного rubert-tiny2: +0,036. Именно они вошли в финальную модель.");
  const im = pic(s, "text.png", L, TOP - 0.05, W, 3.65, "Текстовые модели: отдельно и как добавка к таблице");
  const y = TOP + im.h + 0.2, h = 7.0 - 0.18 - y, cw = (W - 0.4) / 3;
  card(s, L, y, cw, h, "Transfer learning", "Три замороженных кодировщика и логистическая регрессия поверх. Сами по себе слабее TF-IDF: названия лотов короткие и шаблонные.", { size: 11.5 });
  card(s, L + cw + 0.2, y, cw, h, "Fine-tuning rubert-tiny2", "Дообучение всей модели: lr 5·10⁻⁵, батч 64, 2 эпохи, AdamW, взвешенная кросс-энтропия. Лучшая на незнакомых названиях: 0,228 против 0,200.", { size: 11.5, headColor: C.accent1 });
  card(s, L + 2 * (cw + 0.2), y, cw, h, "Текст вместе с таблицей", "Лучше всего — 32 компоненты замороженного rubert-tiny2 в LightGBM: +0,036. Они и вошли в финальную модель.", { size: 11.5, dark: true });
}

// ---------- 6. LLM + QLoRA ----------
{
  const s = content("LLM с QLoRA: слабее бустинга, но полезна в ансамбле",
    "Критерий 5. Qwen2.5-1.5B читает короткую русскую карточку закупки: те же 20 признаков, история, цена и названия лотов — и выдаёт вероятность срыва. Модель сжата до 4 бит, обучаются только адаптеры LoRA — 1,2 % параметров. Полное дообучение не поместилось бы в видеокарту T4. Результат 0,322 — выше любой модели на 20 признаках, но ниже LightGBM с той же информацией на тех же 40 тысячах строк. Зато в ансамбле с бустингом LLM добавляет +0,008, и интервал не включает ноль: ошибки у моделей разные.");
  const lw = 5.7;
  txt(s, "Конфигурация", { x: L, y: TOP, w: lw, h: 0.32, fontSize: 14, bold: true, color: C.text1 });
  const k = (t) => ({ text: t, options: { color: MUTED } });
  const v = (t) => ({ text: t, options: { bold: true } });
  table(s, [
    [k("Базовая модель"), v("Qwen2.5-1.5B + голова классификации")],
    [k("Квантование"), v("4 бита NF4, двойное квантование")],
    [k("LoRA: r / α / dropout"), v("16 / 32 / 0,05")],
    [k("Где адаптеры"), v("все 7 линейных проекций слоя")],
    [k("Обучаемые параметры"), v("18,5 млн из 1,5 млрд (1,2 %)")],
    [k("Обучение"), v("40 тыс. карточек, 1 эпоха, lr 2·10⁻⁴")],
    [k("Видеокарта"), v("Kaggle T4: 2,2 ч обучение, 1,3 ч оценка")],
  ], { x: L, y: TOP + 0.45, w: lw, colW: [2.05, lw - 2.05], rowH: 0.4, fill: { color: "F7FAFC" } });
  card(s, L, TOP + 3.55, lw, 1.45, "Почему не полное дообучение", "Полное дообучение 1,5 млрд параметров требует ≈ 25 ГБ памяти и не помещается в T4 (16 ГБ). QLoRA — пик 2,7 ГБ, адаптер ≈ 70 МБ.", { size: 11.5 });
  const x = 6.75, w = R - x;
  pic(s, "llm.png", x, TOP, w, 3.45, "LLM против LightGBM и их ансамбль");
  card(s, x, TOP + 3.55, w, 1.45, null, "На тех же 40 тыс. строк LightGBM выше на 0,011 [−0,024; 0,000]. В ансамбле LLM добавляет +0,008 [+0,005; +0,012]: ошибки моделей разные.", { dark: true, size: 12 });
}

// ---------- 7. one table ----------
{
  const s = content("Все модели на одном test",
    "Критерий 7. Все модели оценены на одних и тех же 38 981 закупке test. Интервал — бутстрэп. Каждая строка отвечает на свой вопрос: регрессия — точка отсчёта, CatBoost — лучшая модель на старых признаках, дообученный rubert-tiny2 — сколько в одном тексте, LLM — может ли языковая модель заменить бустинг. Финальная модель — CatBoost на 61 признаке: 0,364. P@1 % у неё 0,715: среди 1 % самых рискованных закупок 72 % действительно сорвались. Это примерно 280 найденных срывов из 390 проверок против 196 у мидтерма.");
  const c = (t, o = {}) => ({ text: t, options: o });
  const r = (o = {}) => ({ align: "right", ...o });
  const rows = [
    ["Без модели (доля срывов)", "—", "0,066", "0,500", "0,057"],
    ["Лог. регрессия · мидтерм", "20 признаков", "0,262 [0,246; 0,280]", "0,792", "0,504"],
    ["CatBoost после подбора", "20 признаков", "0,311 [0,293; 0,331]", "0,807", "0,573"],
    ["Дообученный rubert-tiny2", "только названия лотов", "0,156 [0,146; 0,167]", "0,697", "0,242"],
    ["Qwen2.5-1.5B + QLoRA", "карточка закупки", "0,322 [0,302; 0,342]", "0,808", "0,594"],
    ["LightGBM + текст", "20 признаков + текст", "0,347 [0,327; 0,368]", "0,822", "0,671"],
    ["Финальная модель · CatBoost", "61 признак", "0,364 [0,343; 0,385]", "0,827", "0,715"],
    ["Финальная + LLM", "61 признак + LLM", "0,368 [0,347; 0,390]", "0,826", "0,715"],
  ];
  const fin = { bold: true, fill: { color: "FED7D7" } };
  const body = rows.map((row, i) => row.map((t, j) => c(t, { ...(j >= 2 ? r() : {}), ...(i === 6 ? fin : {}) })));
  const tw = 8.55;
  table(s, [[c("Модель", HDR), c("Что видит", HDR), c("PR-AUC [95 %]", r(HDR)), c("ROC-AUC", r(HDR)), c("P@1 %", r(HDR))], ...body],
    { x: L, y: TOP, w: tw, colW: [2.75, 2.2, 1.9, 0.85, 0.85], rowH: 0.5, fontSize: 12 });
  caption(s, "Полная таблица из 25 строк — в ноутбуке 07 и results/a3_07/all_models_test.csv.", L, TOP + 9 * 0.5 + 0.15, tw, 0.35);
  const x = L + tw + 0.35, w = R - x;
  s.addShape(pres.shapes.RECTANGLE, { x, y: TOP, w, h: 2.7, fill: { color: C.text2 }, line: { type: "none" }, objectName: "callout" });
  txt(s, "≈ 280", { x: x + 0.22, y: TOP + 0.2, w: w - 0.4, h: 0.8, fontSize: 40, bold: true, color: C.background1 });
  txt(s, "срывов найдёт проверка 1 % закупок (≈ 390 за две недели) по финальной модели", { x: x + 0.22, y: TOP + 1.05, w: w - 0.4, h: 0.9, fontSize: 12.5, color: C.background1 });
  txt(s, "мидтерм ≈ 196 · случайный выбор ≈ 26", { x: x + 0.22, y: TOP + 2.05, w: w - 0.4, h: 0.5, fontSize: 12, color: ON_DARK });
  card(s, x, TOP + 2.9, w, 1.6, "Как читать", "P@1 % — доля срывов среди 1 % закупок с наибольшей оценкой. Интервал — бутстрэп, 1 000 перевыборок.", { size: 11.5 });
}

// ---------- 8. ablations and SHAP ----------
{
  const s = content("Почти весь новый сигнал — текст лотов",
    "Критерий 8: откуда прирост. Слева абляции: убираем одну группу признаков и переобучаем модель. Без текста качество падает на 0,025, и интервал не включает ноль. История «заказчик–поставщик» и цена к рынку поверх текста почти ничего не дают. Справа SHAP: сумма и способ закупки по-прежнему главные. Доля побед победителя стоит на четвёртом месте, но абляция показывает, что без неё модель берёт ту же информацию из других признаков. Важный признак не обязательно незаменимый.");
  const cw = (W - 0.4) / 2;
  txt(s, "Абляции: убираем группу и переобучаем", { x: L, y: TOP - 0.05, w: cw, h: 0.32, fontSize: 14, bold: true, color: C.text1 });
  txt(s, "SHAP: что двигает предсказания", { x: L + cw + 0.4, y: TOP - 0.05, w: cw, h: 0.32, fontSize: 14, bold: true, color: C.text1 });
  const a = pic(s, "ablation.png", L, TOP + 0.38, cw, 3.75, "Абляции групп признаков");
  const b = pic(s, "shap.png", L + cw + 0.4, TOP + 0.38, cw, 3.75, "Средний модуль SHAP по признакам");
  const y = TOP + 0.38 + Math.max(a.h, b.h) + 0.25, h = Math.min(1.25, 7.0 - 0.18 - y);
  card(s, L, y, cw, h, "Абляции", "Без текста −0,025: интервал не включает ноль. История и цена поверх текста ≈ 0, хотя по отдельности предсказательны.", { size: 11.5 });
  card(s, L + cw + 0.4, y, cw, h, "Важный ≠ незаменимый", "Доля побед победителя — 4-я по SHAP, но без неё модель берёт ту же информацию из других признаков. Сумма и способ закупки, как в мидтерме, — главные.", { size: 11.5 });
}

// ---------- 9. leakage case ----------
{
  const s = content("Случай утечки: «победитель» из таблицы договоров",
    "Это находка по ходу работы. В ранней версии ресёрча признаки активности победителя давали +0,025 и выглядели третьим по важности источником. Победителя я брал из таблицы договоров. Проверка верха списка показала проблему: если победитель отказывается подписывать, договор уходит другому участнику или заключается вне конкурса. Подписант договора не подавал заявку в 27 % успешных и в 63 % сорвавшихся закупок — то есть кто подписал, это уже часть исхода. С победителем из объявления или по минимальной цене прирост исчезает. Урок: слишком хороший признак надо проследить до момента, когда его значение становится известно.");
  const im = pic(s, "leak.png", L, TOP, 6.5, 4.6, "Прирост от признаков победителя при трёх определениях победителя");
  caption(s, "Признаки активности победителя (заявки за 30 дней, ИП или нет) при трёх определениях победителя. Красные — с утечкой.", L, TOP + im.h + 0.15, 6.5, 0.6);
  const x = 7.5, w = R - x;
  s.addShape(pres.shapes.RECTANGLE, { x, y: TOP, w, h: 1.85, fill: { color: C.background2 }, line: { type: "none" }, objectName: "stat leak" });
  txt(s, "27 % и 63 %", { x: x + 0.22, y: TOP + 0.15, w: w - 0.4, h: 0.7, fontSize: 32, bold: true, color: C.accent1 });
  txt(s, "закупок, где подписант договора даже не подавал заявку: среди успешных и среди сорвавшихся", { x: x + 0.22, y: TOP + 0.9, w: w - 0.4, h: 0.8, fontSize: 12, color: C.text1 });
  card(s, x, TOP + 2.05, w, 1.55, "Почему это утечка", "Если победитель отказался подписывать, договор уходит другому участнику или заключается вне конкурса. Кто подписал — уже часть исхода.", { size: 11.5 });
  card(s, x, TOP + 3.8, w, 1.6, "Урок", "Слишком сильный признак — проверить верх списка вручную и проследить, когда значение становится известно. Без утечки прирост равен нулю.", { dark: true, size: 11.5 });
}

// ---------- 10. segments ----------
{
  const s = content("Где модель помогает больше всего",
    "Критерий 8, результаты по сегментам. PR-AUC зависит от доли срывов, поэтому показываю подъём — PR-AUC, делённый на долю срывов в сегменте. Единица — это случайный выбор. Больше всего выиграли закупки из одного источника после несостоявшейся закупки: подъём вырос с 2,2 до 8,0. Самые дешёвые закупки — слабое место мидтерма — тоже заметно лучше. Конкурсы с балльной системой остаются трудными: модель там почти случайна, но таких закупок в test всего 624.");
  pic(s, "segments.png", L, TOP, 7.1, 5.4, "Подъём по сегментам: мидтерм и финальная модель");
  const x = 8.0, w = R - x;
  step(s, 1, x, TOP + 0.05, w, "Из одного источника после несостоявшейся", "подъём 2,2 → 8,0: срывов здесь всего 2 %, и мидтерм их почти не различал", { h: 1.1 });
  step(s, 2, x, TOP + 1.3, w, "Самые дешёвые закупки", "2,3 → 3,6: слабое место мидтерма — при рабочем пороге он не находил там ни одного срыва", { h: 1.1 });
  step(s, 3, x, TOP + 2.55, w, "Конкурсы с балльной системой", "≈ 1,2: почти случайно; всего 624 закупки в test — трудный сегмент", { h: 1.1, badge: C.accent5 });
  card(s, x, TOP + 3.85, w, 1.45, "Как читать", "Подъём = PR-AUC / доля срывов в сегменте; 1 — случайный выбор. Сегменты с разной долей срывов нельзя сравнивать по самому PR-AUC.", { size: 11.5 });
}

// ---------- 11. conclusions ----------
{
  const s = pres.addSlide({ masterName: "CLOSING" });
  s.addText("Выводы и ограничения", { placeholder: "title" });
  label(s, "Что показал A3", L, TOP, 6.9);
  const items = [
    ["0,364", "PR-AUC финальной модели на test (мидтерм 0,262); ROC-AUC 0,827; вероятности откалиброваны"],
    ["+0,05 ×2", "дали сильная модель и новая информация; подбор гиперпараметров почти ничего"],
    ["0,228", "у дообученного rubert-tiny2 на незнакомых названиях — лучше TF-IDF; LLM полезна в ансамбле"],
    ["1 утечка", "найдена и исправлена: победитель из договора — это уже исход"],
  ];
  items.forEach(([big, t], i) => {
    const y = TOP + 0.45 + i * 1.22;
    txt(s, big, { x: L, y, w: 2.15, h: 0.9, fontSize: 28, bold: true, color: C.background1, valign: "middle" });
    txt(s, t, { x: L + 2.3, y, w: 4.6, h: 0.9, fontSize: 13, color: ON_DARK, valign: "middle" });
  });
  const x = 7.95, w = R - x;
  s.addShape(pres.shapes.RECTANGLE, { x, y: TOP, w, h: 5.35, fill: { color: C.background1, transparency: 91 }, line: { type: "none" }, objectName: "limits" });
  label(s, "Ограничения и воспроизводимость", x + 0.25, TOP + 0.25, w - 0.5, ON_DARK);
  const lim = [
    ["Один квартал данных", "декабрьская спешка с бюджетом модели не знакома"],
    ["Метка — статус на портале", "включает расторжения по соглашению сторон"],
    ["LLM на 40 тыс. строк", "больше не позволило время GPU"],
    ["Воспроизводимость", "зерно 42; ноутбуки из build_*.py; данные — релиз data-v1; LLM — скрипт для Kaggle"],
  ];
  lim.forEach(([h, b], i) => step(s, i + 1, x + 0.25, TOP + 0.75 + i * 1.12, w - 0.5, h, b, { color: C.background1, muted: ON_DARK, h: 1.0 }));
  s.addNotes("Итог. Финальная модель — CatBoost на 61 признаке, PR-AUC 0,364 против 0,262 у мидтерма. Прирост поровну дали сильная модель и новая информация, подбор настроек почти ничего. Из текстовых подходов дообучение лучше всего понимает незнакомые названия, а LLM полезна только вместе с бустингом. По ходу работы нашёл и исправил утечку. Ограничения: один квартал данных, метка включает мягкие расторжения, LLM обучена на части данных. Всё воспроизводится: зерно 42, ноутбуки собираются скриптами, данные лежат в релизе.");
}

// ---------- write + theme colors ----------
async function applyTheme(file) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const part = "ppt/theme/theme1.xml";
  const slots = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
  const scheme = `<a:clrScheme name="${THEME.name}">` + slots.map((k) => `<a:${k}><a:srgbClr val="${THEME.colors[k]}"/></a:${k}>`).join("") + "</a:clrScheme>";
  const xml = (await zip.file(part).async("string"))
    .replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, head) => `${head}${THEME.name}"`);
  zip.file(part, xml);
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

pres.writeFile({ fileName: OUT }).then(async () => {
  await applyTheme(OUT);
  console.log("written", OUT);
});
