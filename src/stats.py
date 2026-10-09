"""
src/stats.py
------------
Analytical computation module for IPL Cricket Player Performance Analytics.
Computes career and season metrics (total runs, batting average, strike rate),
rankings (top N batsmen overall & per season), multi-player season comparisons,
and recent match form (last N matches).
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Dict, Any


def load_player_match_data(csv_path: str = "data/player_match.csv") -> pd.DataFrame:
    """Loads and returns the preprocessed player-match dataset."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def calculate_player_overall_stats(
    df: pd.DataFrame, player_name: str
) -> Dict[str, Any]:
    """
    Computes career-level summary statistics for a given player.

    Metrics calculated:
        - matches: Total unique matches played
        - innings: Total innings batted
        - runs: Total runs scored
        - balls_faced: Total deliveries faced (excluding wides)
        - dismissals: Number of times out
        - not_outs: Number of innings remaining not out
        - highest_score: Highest runs scored in a single match
        - batting_average: runs / dismissals (or total runs if never dismissed)
        - strike_rate: (runs / balls_faced) * 100
        - fours: Total 4s hit
        - sixes: Total 6s hit
        - fifties: Number of innings with 50-99 runs
        - centuries: Number of innings with 100+ runs
    """
    pdf = df[df["player"] == player_name]
    if pdf.empty:
        raise ValueError(f"Player '{player_name}' not found in the dataset.")

    matches = int(pdf["match_id"].nunique())
    innings = int(len(pdf))
    runs = int(pdf["runs"].sum())
    balls = int(pdf["balls_faced"].sum())
    outs = int(pdf["dismissed"].sum())
    not_outs = innings - outs
    highest_score = int(pdf["runs"].max())
    fours = int(pdf["fours"].sum())
    sixes = int(pdf["sixes"].sum())
    fifties = int(((pdf["runs"] >= 50) & (pdf["runs"] < 100)).sum())
    centuries = int((pdf["runs"] >= 100).sum())

    # Batting Average: Runs per dismissal
    # If dismissals == 0 (never out), average equals total runs
    batting_avg = round(runs / outs, 2) if outs > 0 else float(runs)

    # Strike Rate: Runs scored per 100 balls faced
    strike_rate = round((runs / balls) * 100.0, 2) if balls > 0 else 0.0

    return {
        "player": player_name,
        "matches": matches,
        "innings": innings,
        "runs": runs,
        "balls_faced": balls,
        "dismissals": outs,
        "not_outs": not_outs,
        "highest_score": highest_score,
        "batting_average": batting_avg,
        "strike_rate": strike_rate,
        "fours": fours,
        "sixes": sixes,
        "fifties": fifties,
        "centuries": centuries,
    }


def calculate_player_season_stats(
    df: pd.DataFrame, player_name: str
) -> pd.DataFrame:
    """
    Computes season-by-season batting performance metrics for a specific player.
    """
    pdf = df[df["player"] == player_name].copy()
    if pdf.empty:
        raise ValueError(f"Player '{player_name}' not found in the dataset.")

    def season_agg(group: pd.DataFrame) -> pd.Series:
        runs = group["runs"].sum()
        balls = group["balls_faced"].sum()
        outs = group["dismissed"].sum()
        innings = len(group)
        not_outs = innings - outs
        avg = round(runs / outs, 2) if outs > 0 else float(runs)
        sr = round((runs / balls) * 100.0, 2) if balls > 0 else 0.0

        return pd.Series({
            "innings": innings,
            "runs": runs,
            "balls_faced": balls,
            "dismissals": outs,
            "not_outs": not_outs,
            "highest_score": group["runs"].max(),
            "batting_average": avg,
            "strike_rate": sr,
            "fours": group["fours"].sum(),
            "sixes": group["sixes"].sum(),
            "fifties": int(((group["runs"] >= 50) & (group["runs"] < 100)).sum()),
            "centuries": int((group["runs"] >= 100).sum()),
        })

    season_df = pdf.groupby("season", as_index=False).apply(
        season_agg, include_groups=False
    )
    season_df.insert(0, "player", player_name)
    season_df.sort_values(by="season", ascending=True, inplace=True)
    season_df.reset_index(drop=True, inplace=True)
    return season_df


def get_top_batsmen(
    df: pd.DataFrame,
    n: int = 10,
    season: Optional[int] = None,
    sort_by: str = "runs",
    min_balls: int = 0,
) -> pd.DataFrame:
    """
    Returns the top N batsmen overall or for a specific season.

    Parameters:
        df: Player-match DataFrame.
        n: Number of top players to return.
        season: Specific IPL season year (e.g. 2024), or None for overall career.
        sort_by: Metric to sort by ('runs', 'batting_average', 'strike_rate').
        min_balls: Minimum balls faced filter (useful for filtering noise in average/SR).
    """
    data = df if season is None else df[df["season"] == season]

    grouped = data.groupby("player", as_index=False).agg(
        matches=("match_id", "nunique"),
        innings=("runs", "count"),
        runs=("runs", "sum"),
        balls_faced=("balls_faced", "sum"),
        dismissals=("dismissed", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        highest_score=("runs", "max"),
    )

    # Calculate average and strike rate
    grouped["batting_average"] = np.where(
        grouped["dismissals"] > 0,
        np.round(grouped["runs"] / grouped["dismissals"], 2),
        grouped["runs"].astype(float),
    )
    grouped["strike_rate"] = np.where(
        grouped["balls_faced"] > 0,
        np.round((grouped["runs"] / grouped["balls_faced"]) * 100.0, 2),
        0.0,
    )

    # Optional filter on minimum balls faced
    if min_balls > 0:
        grouped = grouped[grouped["balls_faced"] >= min_balls]

    if sort_by not in grouped.columns:
        sort_by = "runs"

    grouped.sort_values(by=sort_by, ascending=False, inplace=True)
    top_df = grouped.head(n).reset_index(drop=True)

    if season is not None:
        top_df.insert(0, "season", season)

    return top_df


def compare_players_by_season(
    df: pd.DataFrame, player_names: List[str]
) -> pd.DataFrame:
    """
    Compares two or more players season-by-season.

    Parameters:
        df: Player-match DataFrame.
        player_names: List of player names (e.g. ['V Kohli', 'RG Sharma', 'MS Dhoni']).

    Returns:
        pd.DataFrame: Merged season-by-season table for comparison and plotting.
    """
    comparison_frames = []
    for player in player_names:
        if player in df["player"].values:
            stats_df = calculate_player_season_stats(df, player)
            comparison_frames.append(stats_df)
        else:
            print(f"Warning: Player '{player}' not found in dataset. Skipping.")

    if not comparison_frames:
        return pd.DataFrame()

    combined_df = pd.concat(comparison_frames, ignore_index=True)
    combined_df.sort_values(by=["season", "player"], ascending=[True, True], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


def get_player_last_n_matches(
    df: pd.DataFrame, player_name: str, n: int = 5
) -> pd.DataFrame:
    """
    Returns the most recent N matches for a player, sorted from most recent to oldest.

    Parameters:
        df: Player-match DataFrame.
        player_name: Name of the player.
        n: Number of recent matches.

    Returns:
        pd.DataFrame: Match-level records for recent form evaluation.
    """
    pdf = df[df["player"] == player_name].copy()
    if pdf.empty:
        raise ValueError(f"Player '{player_name}' not found in the dataset.")

    # Ensure sorted by date descending
    pdf.sort_values(by="date", ascending=False, inplace=True)
    recent = pdf.head(n).copy()

    columns_to_show = [
        "match_id",
        "date",
        "season",
        "batting_team",
        "opponent",
        "venue",
        "runs",
        "balls_faced",
        "fours",
        "sixes",
        "dismissed",
        "strike_rate",
    ]
    return recent[columns_to_show].reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING IPL STATS MODULE WITH: 'V Kohli', 'RG Sharma', 'MS Dhoni'")
    print("=" * 80)

    df_pm = load_player_match_data()
    test_players = ["V Kohli", "RG Sharma", "MS Dhoni"]

    # 1. Career Overall Stats
    print("\n--- 1. OVERALL CAREER STATS ---")
    career_records = [calculate_player_overall_stats(df_pm, p) for p in test_players]
    career_df = pd.DataFrame(career_records)
    print(career_df[[
        "player", "matches", "innings", "runs", "balls_faced",
        "dismissals", "batting_average", "strike_rate", "fours", "sixes", "fifties", "centuries"
    ]].to_string(index=False))

    # 2. Top 5 Batsmen Overall
    print("\n--- 2. TOP 5 BATSMEN OVERALL (ALL TIME) ---")
    top_overall = get_top_batsmen(df_pm, n=5)
    print(top_overall[[
        "player", "matches", "innings", "runs", "balls_faced", "batting_average", "strike_rate"
    ]].to_string(index=False))

    # 3. Top 5 Batsmen in IPL 2024 Season
    print("\n--- 3. TOP 5 BATSMEN IN IPL 2024 ---")
    top_2024 = get_top_batsmen(df_pm, n=5, season=2024)
    print(top_2024[[
        "season", "player", "innings", "runs", "balls_faced", "batting_average", "strike_rate"
    ]].to_string(index=False))

    # 4. Compare Players by Season (Sample: 2023 - 2024)
    print("\n--- 4. SEASON COMPARISON (Sample: Seasons 2023 - 2024) ---")
    comp_df = compare_players_by_season(df_pm, test_players)
    sample_comp = comp_df[comp_df["season"].isin([2023, 2024])]
    print(sample_comp[[
        "season", "player", "innings", "runs", "batting_average", "strike_rate"
    ]].to_string(index=False))

    # 5. Last 5 Matches for each player
    print("\n--- 5. LAST 5 MATCHES FORM ---")
    for p in test_players:
        print(f"\nLast 5 matches for {p}:")
        recent_df = get_player_last_n_matches(df_pm, p, n=5)
        print(recent_df[[
            "date", "opponent", "runs", "balls_faced", "fours", "sixes", "dismissed", "strike_rate"
        ]].to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
