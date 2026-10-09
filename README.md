# 🏏 Cricket Player Performance Analytics and Runs Prediction System

An academic project analyzing 18 seasons of Indian Premier League (IPL 2008–2025) data using **Streamlit**, **Pandas**, **Plotly**, and **Scikit-learn** (`RandomForestRegressor`).

---

## 🚀 Quickstart Guide

### 1. Activate Environment & Install Dependencies
```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/Mac

# Install required libraries
pip install -r requirements.txt
```

### 2. Prepare Data (One-Time Execution)
To generate the cleaned batter-match level dataset:
```bash
python src/data_prep.py
```
*Output: `data/player_match.csv` (17,638 rows, 13 features, 0 duplicates).*

### 3. Train Machine Learning Model
To train the zero-data-leakage `RandomForestRegressor` and evaluate against baselines:
```bash
python src/model.py
```
*Outputs: `data/runs_model.joblib` and `data/test_predictions.csv`.*

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

---

## 📊 Streamlit Dashboard Structure

### 1. 📊 Overview Page
- **Macro KPI Cards:** Total Matches (1,169), Unique Players (704), Total Runs (355,373), and Seasons (2008–2025).
- **Season Trend Chart:** Plotly interactive area/line chart depicting the evolution of total runs scored across all 18 seasons.
- **Top 10 Leaderboard:** Filterable leaderboard (All-Time and Season-wise Orange Cap contenders) with an interactive horizontal bar chart.

### 2. 👤 Player Analytics Page
- **Career Summary Cards:** Innings, Total Runs, Batting Average, Strike Rate, 50s/100s, and Boundaries (4s/6s).
- **Season Breakdowns:** Visualizes runs scored per season and year-by-year strike rate / average progression.
- **Recent Form (Last 10 Matches):** Tabular match log and a line chart featuring a 3-match rolling average trend line.
- **Head-to-Head Comparison:** Select any two players for side-by-side metric comparison and multi-season trend tracking.

### 3. 🎯 Runs Prediction Page
- **Model Evaluation Cards:** Test-set comparison of the Random Forest model against two baselines (Global Mean and 5-Match Moving Average).
- **Upcoming Fixture Predictor:** Input any eligible batter, opposition franchise, and venue to generate expected runs and a 68% confidence interval.
- **Historical Backtesting (IPL 2025):** Inspect hold-out test predictions with actual vs. predicted runs and historical innings charts with annotated prediction markers.

---

## 🧠 Machine Learning Methodology

### Zero Data Leakage Feature Design
For every player and match $i$, features are strictly constructed from matches $0 \dots i-1$:
1. `prev_5_avg_runs`: Mean runs in the past 5 innings.
2. `prev_5_strike_rate`: Combined strike rate across the past 5 innings.
3. `career_batting_avg_before`: Career batting average prior to the match.
4. `matches_played_before`: Total match experience prior to the match.
5. `opponent`: One-hot encoded opposition bowling team.
6. `venue`: One-hot encoded match stadium.

### Chronological Season Split
- **Training Set:** IPL 2008 – 2023 ($12,825$ samples)
- **Validation Set:** IPL 2024 ($989$ samples)
- **Test Set:** IPL 2025 ($1,017$ samples)

### Evaluation on Test Set (IPL 2025)
| Model / Baseline | MAE (Runs) | RMSE (Runs) | Strategy |
| :--- | :---: | :---: | :--- |
| **Baseline 1: Global Mean** | 18.07 | 23.35 | Constant train mean ($\approx 21.1$) |
| **Baseline 2: Prev 5-Match Avg** | 16.79 | 22.87 | Moving average form |
| **RandomForestRegressor (ML)** | **16.42** | **21.84** | Recent form + Career level + Opponent + Venue |

> **Academic Note on Model Performance:** In T20 cricket, individual scores have extreme variance (ducks vs rapid 80s). The ML model outperforms the moving average baseline by **1.03 RMSE**, demonstrating genuine predictive signal from match context while respecting the inherent statistical floor of single-innings sports outcomes.
