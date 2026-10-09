"""
app.py
------
Cricket Player Performance Analytics and Runs Prediction System.
Academic Project built with Streamlit, Pandas, Plotly, and Scikit-learn.

Pages:
    1. Overview: Macro-level IPL analytics (matches, players, runs, top scorers, season trends).
    2. Player Analytics: Micro-level player evaluation (career KPIs, season breakdowns, last 10 matches, player comparison).
    3. Runs Prediction: Machine learning inference (next match prediction, test set backtesting, baseline comparisons).
"""

import os
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.stats import (
    calculate_player_overall_stats,
    calculate_player_season_stats,
    get_top_batsmen,
    compare_players_by_season,
    get_player_last_n_matches,
)
from src.model import predict_player_match_runs

# ---------------------------------------------------------
# STREAMLIT APP CONFIGURATION & THEMING
# ---------------------------------------------------------
st.set_page_config(
    page_title="IPL Analytics & Runs Prediction",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern glassmorphism aesthetic and refined typography
CUSTOM_CSS = """
<style>
    /* Global imports & styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Metric card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(245, 158, 11, 0.4);
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-subtitle {
        color: #F59E0B;
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-top: 18px;
        margin-bottom: 12px;
        border-bottom: 2px solid rgba(245, 158, 11, 0.3);
        padding-bottom: 6px;
    }

    /* Info callout */
    .custom-callout {
        background: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0EA5E9;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #E2E8F0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# CACHED DATA LOADERS
# ---------------------------------------------------------
@st.cache_data
def load_main_dataset():
    """Loads and caches the cleaned player-match level dataset."""
    df = pd.read_csv("data/player_match.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_test_predictions():
    """Loads and enriches the test set predictions (IPL 2025)."""
    df_pm = load_main_dataset()
    df_pred = pd.read_csv("data/test_predictions.csv")
    df_pred["date"] = pd.to_datetime(df_pred["date"])

    # Merge with player_match to enrich with match metadata (opponent, venue, match_id)
    enriched = df_pred.merge(
        df_pm[["player", "date", "match_id", "opponent", "venue", "balls_faced", "strike_rate"]],
        on=["player", "date"],
        how="inner",
    )
    # Calculate absolute error
    enriched["error"] = np.abs(enriched["predicted"] - enriched["actual"]).round(1)
    enriched.sort_values(by="date", ascending=False, inplace=True)
    return enriched


@st.cache_resource
def load_ml_pipeline():
    """Loads the trained RandomForestRegressor pipeline."""
    model_path = "data/runs_model.joblib"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# Helper Plotly styling
def apply_plotly_theme(fig):
    """Applies a consistent dark theme and polished styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.06)")
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.06)")
    return fig


# ---------------------------------------------------------
# PAGE 1: OVERVIEW
# ---------------------------------------------------------
def render_overview(df: pd.DataFrame):
    st.title("🏏 IPL Performance Overview")
    st.markdown(
        "Macro-level analytics covering the Indian Premier League from **2008 to 2025**."
    )

    # 1. Macro KPIs
    total_matches = df["match_id"].nunique()
    total_players = df["player"].nunique()
    total_runs = int(df["runs"].sum())
    seasons_list = sorted(df["season"].unique())
    total_seasons = len(seasons_list)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Matches</div>
            <div class="metric-value">{total_matches:,}</div>
            <div class="metric-subtitle">18 Complete Editions</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Players</div>
            <div class="metric-value">{total_players:,}</div>
            <div class="metric-subtitle">Unique Batters</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Runs Scored</div>
            <div class="metric-value">{total_runs:,}</div>
            <div class="metric-subtitle">Across All Innings</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Seasons Covered</div>
            <div class="metric-value">{seasons_list[0]}–{seasons_list[-1]}</div>
            <div class="metric-subtitle">{total_seasons} Years of T20 Cricket</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Season-wise Total Runs Trend
    st.markdown('<div class="section-header">📈 Season-wise Total Runs & Scoring Trend</div>', unsafe_allow_html=True)
    
    season_agg = df.groupby("season", as_index=False).agg(
        total_runs=("runs", "sum"),
        matches=("match_id", "nunique"),
        avg_runs_per_match=("runs", lambda x: round(x.sum() / df.loc[x.index, "match_id"].nunique(), 1)),
        sixes=("sixes", "sum"),
        fours=("fours", "sum"),
    )

    fig_season = go.Figure()
    fig_season.add_trace(go.Scatter(
        x=season_agg["season"],
        y=season_agg["total_runs"],
        mode="lines+markers",
        name="Total Runs",
        line=dict(color="#F59E0B", width=3),
        marker=dict(size=8, color="#FBBF24"),
        fill="tozeroy",
        fillcolor="rgba(245, 158, 11, 0.12)",
        hovertemplate="<b>Season %{x}</b><br>Total Runs: %{y:,}<extra></extra>",
    ))
    fig_season.update_layout(
        title="Total IPL Runs Scored Per Season (2008 – 2025)",
        xaxis_title="Season Year",
        yaxis_title="Total Runs",
        height=380,
    )
    apply_plotly_theme(fig_season)
    st.plotly_chart(fig_season, use_container_width=True)

    # 3. Top 10 Run Scorers (Overall & Per Season Filter)
    st.markdown('<div class="section-header">🏆 Top 10 All-Time Run Scorers</div>', unsafe_allow_html=True)
    
    col_filter, col_chart = st.columns([1, 3])
    with col_filter:
        st.markdown("**Filter Leaderboard**")
        season_option = st.selectbox(
            "Select Season",
            options=["All Time"] + [str(s) for s in reversed(seasons_list)],
            index=0,
            help="Filter between all-time leaders and specific season Orange Cap contenders.",
        )
        selected_season = None if season_option == "All Time" else int(season_option)
        top10_df = get_top_batsmen(df, n=10, season=selected_season)
        
        st.dataframe(
            top10_df[["player", "runs", "batting_average", "strike_rate"]],
            column_config={
                "player": "Player",
                "runs": st.column_config.NumberColumn("Runs", format="%d"),
                "batting_average": st.column_config.NumberColumn("Avg", format="%.2f"),
                "strike_rate": st.column_config.NumberColumn("SR", format="%.1f"),
            },
            hide_index=True,
            use_container_width=True,
            height=380,
        )

    with col_chart:
        # Horizontal Bar Chart for Top 10
        sorted_top10 = top10_df.sort_values(by="runs", ascending=True)
        fig_top = px.bar(
            sorted_top10,
            x="runs",
            y="player",
            orientation="h",
            text="runs",
            color="runs",
            color_continuous_scale="Viridis",
            title=f"Top 10 Run Scorers ({season_option})",
        )
        fig_top.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_top.update_layout(
            xaxis_title="Runs Scored",
            yaxis_title="",
            height=400,
            coloraxis_showscale=False,
        )
        apply_plotly_theme(fig_top)
        st.plotly_chart(fig_top, use_container_width=True)


# ---------------------------------------------------------
# PAGE 2: PLAYER ANALYTICS
# ---------------------------------------------------------
def render_player_analytics(df: pd.DataFrame):
    st.title("👤 Player Performance Analytics")
    st.markdown("Detailed batting KPIs, historical progression, recent form, and head-to-head comparisons.")

    # Player Selection: sort players by total runs descending so star players appear at top
    player_totals = df.groupby("player")["runs"].sum().sort_values(ascending=False)
    all_players = player_totals.index.tolist()

    default_idx = all_players.index("V Kohli") if "V Kohli" in all_players else 0

    selected_player = st.selectbox(
        "Select Primary Player",
        options=all_players,
        index=default_idx,
        help="Search or select any batter from IPL 2008–2025.",
    )

    if not selected_player:
        return

    # 1. Career Stats Cards
    c_stats = calculate_player_overall_stats(df, selected_player)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Innings</div>
            <div class="metric-value">{c_stats['innings']}</div>
            <div class="metric-subtitle">{c_stats['matches']} Matches</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Runs</div>
            <div class="metric-value">{c_stats['runs']:,}</div>
            <div class="metric-subtitle">High: {c_stats['highest_score']}</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Batting Avg</div>
            <div class="metric-value">{c_stats['batting_average']:.2f}</div>
            <div class="metric-subtitle">{c_stats['not_outs']} Not Outs</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Strike Rate</div>
            <div class="metric-value">{c_stats['strike_rate']:.1f}</div>
            <div class="metric-subtitle">{c_stats['balls_faced']:,} Balls</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">50s / 100s</div>
            <div class="metric-value">{c_stats['fifties']} / {c_stats['centuries']}</div>
            <div class="metric-subtitle">Milestones</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Boundaries</div>
            <div class="metric-value">{c_stats['fours']} / {c_stats['sixes']}</div>
            <div class="metric-subtitle">4s / 6s</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Season-wise Runs & Strike Rate Charts
    st.markdown(f'<div class="section-header">📊 Season Breakdown for {selected_player}</div>', unsafe_allow_html=True)
    p_seasons = calculate_player_season_stats(df, selected_player)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        fig_s_runs = px.bar(
            p_seasons,
            x="season",
            y="runs",
            text="runs",
            title=f"Runs Scored by Season ({selected_player})",
            color="runs",
            color_continuous_scale="Purples",
        )
        fig_s_runs.update_traces(textposition="outside")
        fig_s_runs.update_layout(xaxis_title="Season", yaxis_title="Runs", height=350, coloraxis_showscale=False)
        apply_plotly_theme(fig_s_runs)
        st.plotly_chart(fig_s_runs, use_container_width=True)

    with col_s2:
        fig_s_sr = go.Figure()
        fig_s_sr.add_trace(go.Scatter(
            x=p_seasons["season"],
            y=p_seasons["strike_rate"],
            mode="lines+markers",
            name="Strike Rate",
            line=dict(color="#10B981", width=3),
            marker=dict(size=8),
        ))
        fig_s_sr.add_trace(go.Scatter(
            x=p_seasons["season"],
            y=p_seasons["batting_average"],
            mode="lines+markers",
            name="Batting Average",
            line=dict(color="#38BDF8", width=3, dash="dot"),
            marker=dict(size=8),
        ))
        fig_s_sr.update_layout(
            title=f"Strike Rate & Average Trend ({selected_player})",
            xaxis_title="Season",
            yaxis_title="Metric Value",
            height=350,
        )
        apply_plotly_theme(fig_s_sr)
        st.plotly_chart(fig_s_sr, use_container_width=True)

    # 3. Last 10 Matches Form (Table & Rolling Average Line)
    st.markdown(f'<div class="section-header">🔥 Recent Form: Last 10 Matches</div>', unsafe_allow_html=True)
    recent_10 = get_player_last_n_matches(df, selected_player, n=10)

    # Calculate rolling 3-match average in chronological order
    recent_chronological = recent_10.sort_values(by="date").reset_index(drop=True)
    recent_chronological["rolling_avg"] = recent_chronological["runs"].rolling(window=3, min_periods=1).mean().round(1)

    col_r_chart, col_r_table = st.columns([3, 2])
    with col_r_chart:
        fig_recent = go.Figure()
        # Bar of individual scores
        fig_recent.add_trace(go.Bar(
            x=recent_chronological["date"].dt.strftime("%Y-%m-%d"),
            y=recent_chronological["runs"],
            name="Innings Runs",
            marker_color="#F59E0B",
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Runs: %{y}<extra></extra>",
        ))
        # Rolling average trend line
        fig_recent.add_trace(go.Scatter(
            x=recent_chronological["date"].dt.strftime("%Y-%m-%d"),
            y=recent_chronological["rolling_avg"],
            mode="lines+markers",
            name="3-Match Rolling Avg",
            line=dict(color="#06B6D4", width=3),
            marker=dict(size=6, color="#22D3EE"),
            hovertemplate="<b>Rolling Avg: %{y}</b><extra></extra>",
        ))
        fig_recent.update_layout(
            title="Recent Innings Scores & Rolling Average Line",
            xaxis_title="Match Date",
            yaxis_title="Runs Scored",
            height=360,
        )
        apply_plotly_theme(fig_recent)
        st.plotly_chart(fig_recent, use_container_width=True)

    with col_r_table:
        st.markdown("**Last 10 Matches Table**")
        st.dataframe(
            recent_10[["date", "opponent", "runs", "balls_faced", "strike_rate", "dismissed"]],
            column_config={
                "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "opponent": "Opponent",
                "runs": st.column_config.NumberColumn("Runs", format="%d"),
                "balls_faced": st.column_config.NumberColumn("Balls", format="%d"),
                "strike_rate": st.column_config.NumberColumn("SR", format="%.1f"),
                "dismissed": st.column_config.CheckboxColumn("Out?"),
            },
            hide_index=True,
            use_container_width=True,
            height=330,
        )

    # 4. Compare with Another Player Section
    st.markdown('<div class="section-header">⚔️ Head-to-Head Player Comparison</div>', unsafe_allow_html=True)
    
    # Filter out primary player
    comparison_candidates = [p for p in all_players if p != selected_player]
    default_p2 = "RG Sharma" if "RG Sharma" in comparison_candidates else comparison_candidates[0]
    
    selected_p2 = st.selectbox(
        "Select Competitor to Compare",
        options=comparison_candidates,
        index=comparison_candidates.index(default_p2) if default_p2 in comparison_candidates else 0,
    )

    if selected_p2:
        c_stats_p2 = calculate_player_overall_stats(df, selected_p2)
        
        # Comparison Metrics Table
        comp_metrics = pd.DataFrame([
            {"Metric": "Innings Batted", selected_player: c_stats["innings"], selected_p2: c_stats_p2["innings"]},
            {"Metric": "Total Runs", selected_player: f"{c_stats['runs']:,}", selected_p2: f"{c_stats_p2['runs']:,}"},
            {"Metric": "Batting Average", selected_player: f"{c_stats['batting_average']:.2f}", selected_p2: f"{c_stats_p2['batting_average']:.2f}"},
            {"Metric": "Strike Rate", selected_player: f"{c_stats['strike_rate']:.1f}", selected_p2: f"{c_stats_p2['strike_rate']:.1f}"},
            {"Metric": "50s / 100s", selected_player: f"{c_stats['fifties']} / {c_stats['centuries']}", selected_p2: f"{c_stats_p2['fifties']} / {c_stats_p2['centuries']}"},
            {"Metric": "Fours / Sixes", selected_player: f"{c_stats['fours']} / {c_stats['sixes']}", selected_p2: f"{c_stats_p2['fours']} / {c_stats_p2['sixes']}"},
        ])

        col_c_tab, col_c_chart = st.columns([2, 3])
        with col_c_tab:
            st.markdown(f"**Career Overview: {selected_player} vs {selected_p2}**")
            st.dataframe(comp_metrics, hide_index=True, use_container_width=True, height=260)

        with col_c_chart:
            # Multi-player season progression
            dual_season = compare_players_by_season(df, [selected_player, selected_p2])
            fig_compare = px.line(
                dual_season,
                x="season",
                y="runs",
                color="player",
                markers=True,
                title=f"Season-by-Season Runs Comparison",
                color_discrete_map={selected_player: "#F59E0B", selected_p2: "#06B6D4"},
            )
            fig_compare.update_traces(line=dict(width=3), marker=dict(size=8))
            fig_compare.update_layout(xaxis_title="Season", yaxis_title="Runs Scored", height=320)
            apply_plotly_theme(fig_compare)
            st.plotly_chart(fig_compare, use_container_width=True)


# ---------------------------------------------------------
# PAGE 3: RUNS PREDICTION
# ---------------------------------------------------------
def render_runs_prediction(df: pd.DataFrame):
    st.title("🎯 Batter Runs Prediction System")
    st.markdown(
        "Machine Learning model (`RandomForestRegressor`) trained on **IPL 2008–2023**, "
        "validated on **2024**, and tested on **IPL 2025** using strictly historical features."
    )

    # 1. Model vs Baselines Comparison Cards
    st.markdown('<div class="section-header">📊 Model Evaluation vs Baselines (Test Season: 2025)</div>', unsafe_allow_html=True)
    
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Baseline 1: Global Mean</div>
            <div class="metric-value">18.07 <span style="font-size:1rem;color:#94A3B8;">MAE</span></div>
            <div class="metric-subtitle">RMSE: 23.35 | Predicts Train Mean (21.1)</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Baseline 2: Prev 5-Match Avg</div>
            <div class="metric-value">16.79 <span style="font-size:1rem;color:#94A3B8;">MAE</span></div>
            <div class="metric-subtitle">RMSE: 22.87 | Moving Average Form</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div class="metric-card" style="border-color: rgba(16, 185, 129, 0.4);">
            <div class="metric-title" style="color: #10B981;">RandomForestRegressor (ML)</div>
            <div class="metric-value" style="color: #10B981;">16.42 <span style="font-size:1rem;color:#A7F3D0;">MAE</span></div>
            <div class="metric-subtitle" style="color: #34D399;">RMSE: 21.84 | Beats Baseline by 1.02 RMSE</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("ℹ️ Academic Explanation: Why does the ML model only slightly beat the baseline?"):
        st.markdown("""
        - **High Inherent Stochasticity in T20 Cricket:** In modern T20 cricket, batters take immediate aggressive risks. A world-class batter can easily get out on ball 1 to a swinging delivery or direct-hit run out (0 runs), or score 80+ if dropped early.
        - **Low Serial Correlation:** Unlike seasonal averages, individual cricket innings have extreme variance. Past 5-game form provides a good anchor, but exact single-game predictions are non-deterministic.
        - **Regression to the Mean:** The model predicts the expected conditional score (~20–35 runs), avoiding wild single-match predictions that would blow up squared error loss.
        - **Real ML Value:** The Random Forest incorporates opponent bowling attack, venue scoring pace, and career baseline vs. short-term form, shaving **1.03 off RMSE** and beating the baseline while honoring the realistic statistical ceiling of sports prediction.
        """)

    # 2. Section A: Next Match Runs Prediction
    st.markdown('<div class="section-header">🔮 Predict Next Match Runs (Upcoming Fixture)</div>', unsafe_allow_html=True)

    # Filter to players with at least 5 matches played
    match_counts = df.groupby("player")["match_id"].count()
    eligible_players = match_counts[match_counts >= 5].index.tolist()
    # Sort by total runs descending
    player_runs = df.groupby("player")["runs"].sum()
    eligible_players = sorted(eligible_players, key=lambda p: player_runs.get(p, 0), reverse=True)

    unique_teams = sorted(df["opponent"].unique())
    unique_venues = sorted(df["venue"].unique())

    col_p, col_opp, col_ven = st.columns(3)
    with col_p:
        pred_player = st.selectbox("Select Batter", eligible_players, index=0)
    with col_opp:
        pred_opp = st.selectbox("Select Opposition Team", unique_teams, index=0)
    with col_ven:
        pred_venue = st.selectbox("Select Match Venue", unique_venues, index=0)

    if st.button("🚀 Generate Runs Prediction", type="primary"):
        try:
            prediction_res = predict_player_match_runs(pred_player, pred_opp, pred_venue)
            
            p_res1, p_res2, p_res3, p_res4 = st.columns(4)
            with p_res1:
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid #F59E0B;">
                    <div class="metric-title" style="color: #FBBF24;">Expected Runs</div>
                    <div class="metric-value" style="color: #F59E0B;">{prediction_res['predicted_runs']}</div>
                    <div class="metric-subtitle">Predicted Score</div>
                </div>
                """, unsafe_allow_html=True)
            with p_res2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Recent 5-Match Avg</div>
                    <div class="metric-value">{prediction_res['prev_5_avg_runs']}</div>
                    <div class="metric-subtitle">Strike Rate: {prediction_res['prev_5_strike_rate']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            with p_res3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Career Batting Avg</div>
                    <div class="metric-value">{prediction_res['career_batting_avg']:.2f}</div>
                    <div class="metric-subtitle">Across {prediction_res['matches_played']} matches</div>
                </div>
                """, unsafe_allow_html=True)
            with p_res4:
                # Approximate 68% prediction interval (+/- 1 MAE)
                lower_bound = max(0, round(prediction_res['predicted_runs'] - 16.4, 0))
                upper_bound = round(prediction_res['predicted_runs'] + 16.4, 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Expected Range (±1 MAE)</div>
                    <div class="metric-value">{int(lower_bound)} – {int(upper_bound)}</div>
                    <div class="metric-subtitle">68% Confidence Band</div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

    # 3. Section B: Backtesting on Test Set Matches (IPL 2025)
    st.markdown('<div class="section-header">🧪 Backtest on Test Set (IPL 2025 Matches)</div>', unsafe_allow_html=True)
    st.markdown("Inspect real historical predictions from the hold-out test set to compare model output vs reality.")

    test_preds_df = load_test_predictions()

    # Search filter by player
    test_players = sorted(test_preds_df["player"].unique())
    selected_test_player = st.selectbox(
        "Filter Test Matches by Player",
        options=["All Players"] + test_players,
        index=test_players.index("V Kohli") + 1 if "V Kohli" in test_players else 0,
    )

    if selected_test_player != "All Players":
        filtered_test_df = test_preds_df[test_preds_df["player"] == selected_test_player].copy()
    else:
        filtered_test_df = test_preds_df.copy()

    # Build match selector labels
    filtered_test_df["label"] = (
        filtered_test_df["player"]
        + " | "
        + filtered_test_df["date"].dt.strftime("%Y-%m-%d")
        + " vs "
        + filtered_test_df["opponent"]
        + " (Actual: "
        + filtered_test_df["actual"].astype(str)
        + " | Pred: "
        + filtered_test_df["predicted"].astype(str)
        + ")"
    )

    match_choice = st.selectbox(
        "Select Specific Test Match to Inspect",
        options=filtered_test_df["label"].tolist(),
        index=0,
    )

    selected_match_row = filtered_test_df[filtered_test_df["label"] == match_choice].iloc[0]

    # Show evaluation cards for this match
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Actual Runs Scored</div>
            <div class="metric-value" style="color: #38BDF8;">{int(selected_match_row['actual'])}</div>
            <div class="metric-subtitle">{selected_match_row['balls_faced']} Balls ({selected_match_row['strike_rate']} SR)</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Runs</div>
            <div class="metric-value" style="color: #F59E0B;">{selected_match_row['predicted']}</div>
            <div class="metric-subtitle">Random Forest Output</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        err = selected_match_row['error']
        err_color = "#10B981" if err <= 10 else ("#FBBF24" if err <= 20 else "#F43F5E")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Absolute Error</div>
            <div class="metric-value" style="color: {err_color};">{err}</div>
            <div class="metric-subtitle">|Actual - Predicted|</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Match Context</div>
            <div class="metric-value" style="font-size:1.1rem;margin-top:6px;">{selected_match_row['opponent']}</div>
            <div class="metric-subtitle">{selected_match_row['venue']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Plot Player's History with Prediction Marked
    target_player = selected_match_row["player"]
    match_date = selected_match_row["date"]
    
    # Get all player matches up to test match date
    player_history = df[df["player"] == target_player].sort_values(by="date").copy()
    # Keep last 15 matches for clean display
    recent_history = player_history.tail(15).copy()

    fig_hist = go.Figure()
    # History line
    fig_hist.add_trace(go.Scatter(
        x=recent_history["date"].dt.strftime("%Y-%m-%d"),
        y=recent_history["runs"],
        mode="lines+markers",
        name="Actual Innings Runs",
        line=dict(color="#38BDF8", width=2.5),
        marker=dict(size=7, color="#0284C7"),
        hovertemplate="<b>Date: %{x}</b><br>Runs: %{y}<extra></extra>",
    ))

    # Actual test point (highlighted)
    test_date_str = match_date.strftime("%Y-%m-%d")
    fig_hist.add_trace(go.Scatter(
        x=[test_date_str],
        y=[selected_match_row["actual"]],
        mode="markers",
        name="Actual Test Score",
        marker=dict(size=14, color="#38BDF8", symbol="diamond", line=dict(width=2, color="#FFFFFF")),
        hovertemplate=f"<b>Actual Test Score</b>: {selected_match_row['actual']} runs<extra></extra>",
    ))

    # Prediction marker
    fig_hist.add_trace(go.Scatter(
        x=[test_date_str],
        y=[selected_match_row["predicted"]],
        mode="markers",
        name="Model Prediction",
        marker=dict(size=15, color="#F59E0B", symbol="star", line=dict(width=2, color="#FFFFFF")),
        hovertemplate=f"<b>Model Prediction</b>: {selected_match_row['predicted']} runs<extra></extra>",
    ))

    fig_hist.update_layout(
        title=f"Recent Innings History for {target_player} & Prediction Comparison",
        xaxis_title="Match Date",
        yaxis_title="Runs Scored",
        height=400,
    )
    apply_plotly_theme(fig_hist)
    st.plotly_chart(fig_hist, use_container_width=True)


# ---------------------------------------------------------
# MAIN APP ENTRY POINT WITH SIDEBAR ROUTING
# ---------------------------------------------------------
def main():
    # Load dataset
    df = load_main_dataset()

    # Sidebar Navigation
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/en/thumb/8/84/Indian_Premier_League_Official_Logo.svg/1200px-Indian_Premier_League_Official_Logo.svg.png",
        width=150,
    )
    st.sidebar.title("IPL Analytics")
    st.sidebar.markdown("Cricket Player Analytics & Runs Prediction System")

    selected_page = st.sidebar.radio(
        "Navigate",
        options=["📊 Overview", "👤 Player Analytics", "🎯 Runs Prediction"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Tech Stack:**
        - Python 3.13
        - Pandas & NumPy
        - Plotly Charts
        - Scikit-learn (RandomForest)
        - Streamlit Dashboard
        """
    )

    # Route to selected page
    if selected_page == "📊 Overview":
        render_overview(df)
    elif selected_page == "👤 Player Analytics":
        render_player_analytics(df)
    elif selected_page == "🎯 Runs Prediction":
        render_runs_prediction(df)


if __name__ == "__main__":
    main()
