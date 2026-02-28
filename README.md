# Dirty Signal Lab 🌈💧

**Dirty Signal Lab** is a public research + engineering showcase that mirrors how quant teams turn messy market data into deployable signals. It demonstrates:

- Data quality triage & repair on dirty tick data
- Microstructure signal research (order-flow imbalance, microprice)
- Predictive feature engineering with decay
- Walk‑forward backtesting with costs & slippage
- Reproducible, automated pipelines (CLI + CI)
- kdb+‑style interfaces (mocked) + C++/R snippets

> Built as a realistic demonstration of what a Quantitative Analyst does in a data‑driven, research‑heavy trading environment.

---

## 🧭 Project map

```
.
├── src/dirty_signal_lab/        # core pipeline
├── data/raw/                    # generated dirty ticks
├── data/processed/              # cleaned + features
├── reports/                     # generated markdown reports
├── notebooks/                   # optional notebooks
├── cpp/                         # C++ acceleration example
├── r/                           # R exploration example
└── .github/workflows/ci.yml     # CI
```

---

## ✅ Quickstart

```bash
# 1) Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) Run pipeline
python -m dirty_signal_lab.cli run \
  --symbol DEMO \
  --n 20000 \
  --seed 7

# 3) Inspect outputs
ls data/processed/
ls reports/
```

---

## 🔬 What the pipeline does

1. **Generate dirty tick data**
   - Duplicates, missing timestamps, out‑of‑order rows
   - Spreads occasionally inverted / negative

2. **Clean & standardize**
   - Sort by time, remove duplicates, fix anomalies
   - Data‑quality report with counts

3. **Microstructure signals**
   - Order‑flow imbalance
   - Microprice
   - Volatility regime proxy

4. **Predictive features + model**
   - Rolling/decayed stats
   - Z‑scored feature stack
   - Ridge‑regularized linear model (walk‑forward split)

5. **Backtest**
   - Model‑driven signal
   - Costs, slippage, turnover controls

6. **Report**
   - Markdown summary + key metrics

---

## 🧪 Model (simple but non‑trivial)
The pipeline now trains a **ridge‑regularized linear model** on a walk‑forward split
(first 70% train, rest predict) using:

- OFI z‑score
- Microprice z‑score
- Volatility regime z‑score
- OFI EMA z‑score

The model score is squashed with `tanh` to keep positions bounded.

---

## 🧠 Why this is interesting (Quant perspective)

- **Dirty data handling** is often the limiting factor in alpha discovery.
- **Microstructure signals** remain under‑appreciated in many academic‑only projects.
- **Walk‑forward validation** + transaction costs model real‑world constraints.
- **Automation** shows readiness for production‑like research workflows.

---

## 📦 kdb+ mock
This repo includes a lightweight **kdb+‑style mock interface** to show how a q‑like data layer might integrate into a Python pipeline.

```python
from dirty_signal_lab.kdb_mock import KdbMock

kdb = KdbMock.from_csv("data/processed/clean_ticks.csv")
subset = kdb.select("ticks", sym="DEMO", start="2026-01-01", end="2026-01-02")
```

---

## ⚡ C++ / R snippets
- **C++**: `cpp/feature_accel.cpp` shows a toy rolling‑mean acceleration pattern.
- **R**: `r/eda.R` performs exploratory analysis and simple plots.

---

## ✅ CI
GitHub Actions runs:
- `pytest`
- basic linting (ruff)

---

## Roadmap
- Add realistic order book simulator
- Add hyper‑parameter search (walk‑forward grid)
- Add q‑style table joins in kdb mock

---

## License
MIT
