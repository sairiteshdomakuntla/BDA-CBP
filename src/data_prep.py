"""
src/data_prep.py
----------------
Data preparation and feature engineering module for IPL Cricket Analytics.
Aggregates ball-by-ball delivery data into a match-by-match batter performance dataset.

Output dataset schema (one row per batter per match):
    - player: Name of the batter
    - match_id: Unique match identifier
    - date: Date of the match (YYYY-MM-DD)
    - season: IPL season year
    - batting_team: Standardized franchise name batting
    - opponent: Standardized franchise name bowling
    - venue: Standardized venue / stadium name
    - runs: Total runs scored by the batter in the match
    - balls_faced: Total deliveries faced (excluding wides)
    - fours: Number of 4s scored
    - sixes: Number of 6s scored
    - dismissed: Boolean indicating whether the batter was out in that match
    - strike_rate: Batting strike rate (runs / balls_faced * 100)
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# STANDARDIZATION DICTIONARIES
# ---------------------------------------------------------
# Handles franchise rebrandings and naming variants across IPL seasons
TEAM_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

# Handles stadium name changes, city suffixes, and spelling variants
VENUE_MAPPING = {
    # Delhi
    "Arun Jaitley Stadium": "Arun Jaitley Stadium, Delhi",
    "Feroz Shah Kotla": "Arun Jaitley Stadium, Delhi",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium, Delhi",
    # Mumbai
    "Brabourne Stadium": "Brabourne Stadium, Mumbai",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium, Mumbai",
    "Dr DY Patil Sports Academy": "Dr DY Patil Sports Academy, Mumbai",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy, Mumbai",
    "Wankhede Stadium": "Wankhede Stadium, Mumbai",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium, Mumbai",
    # Visakhapatnam
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    # Kolkata
    "Eden Gardens": "Eden Gardens, Kolkata",
    "Eden Gardens, Kolkata": "Eden Gardens, Kolkata",
    # Dharamsala
    "Himachal Pradesh Cricket Association Stadium": "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    # Bengaluru
    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium, Bengaluru",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",
    # Chennai
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chepauk, Chennai",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium, Chepauk, Chennai",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium, Chepauk, Chennai",
    # Mullanpur
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur",
    # Pune
    "Maharashtra Cricket Association Stadium": "Maharashtra Cricket Association Stadium, Pune",
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium, Pune",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium, Pune",
    # Ahmedabad
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium, Ahmedabad",
    # Mohali
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium, Mohali",
    # Hyderabad
    "Rajiv Gandhi International Stadium": "Rajiv Gandhi International Stadium, Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium, Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium, Hyderabad",
    # Jaipur
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium, Jaipur",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium, Jaipur",
    # UAE
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium, Abu Dhabi",
    "Sheikh Zayed Stadium": "Sheikh Zayed Stadium, Abu Dhabi",
}


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading and trailing whitespace from column names and string cells."""
    df.columns = df.columns.str.strip()
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
    return df


def build_player_match_dataset(
    matches_csv: str = "data/matches_updated_ipl_upto_2025.csv",
    deliveries_csv: str = "data/deliveries_updated_ipl_upto_2025.csv",
    output_csv: str = "data/player_match.csv",
) -> pd.DataFrame:
    """
    Builds and saves the batter-match level dataset from raw IPL CSV files.

    Parameters:
        matches_csv (str): Path to match-level CSV file.
        deliveries_csv (str): Path to ball-by-ball delivery CSV file.
        output_csv (str): Path to save the processed dataset.

    Returns:
        pd.DataFrame: Cleaned and structured player-match dataframe.
    """
    print("[1/6] Loading datasets...")
    matches_df = pd.read_csv(matches_csv, low_memory=False)
    deliveries_df = pd.read_csv(deliveries_csv, low_memory=False)

    # Clean whitespace
    matches_df = clean_string_columns(matches_df)
    deliveries_df = clean_string_columns(deliveries_df)

    print("[2/6] Parsing dates and standardizing teams & venues...")
    # Parse match dates to datetime and extract 4-digit IPL season year
    matches_df["parsed_date"] = pd.to_datetime(matches_df["date"])
    matches_df["standardized_season"] = matches_df["parsed_date"].dt.year
    matches_df["standardized_venue"] = matches_df["venue"].map(
        lambda v: VENUE_MAPPING.get(v, v)
    )

    # Standardize team names in deliveries
    deliveries_df["batting_team"] = deliveries_df["batting_team"].map(
        lambda t: TEAM_MAPPING.get(t, t)
    )
    deliveries_df["bowling_team"] = deliveries_df["bowling_team"].map(
        lambda t: TEAM_MAPPING.get(t, t)
    )

    # Clean numeric fields in deliveries
    deliveries_df["batsman_runs"] = pd.to_numeric(
        deliveries_df["batsman_runs"], errors="coerce"
    ).fillna(0).astype(int)
    deliveries_df["isWide"] = pd.to_numeric(
        deliveries_df["isWide"], errors="coerce"
    ).fillna(0)

    # In cricket, a ball faced by a batter excludes wides (no-balls count as balls faced)
    deliveries_df["is_ball_faced"] = (deliveries_df["isWide"] == 0).astype(int)
    deliveries_df["is_four"] = (deliveries_df["batsman_runs"] == 4).astype(int)
    deliveries_df["is_six"] = (deliveries_df["batsman_runs"] == 6).astype(int)

    print("[3/6] Aggregating ball-by-ball deliveries per batter per match...")
    # Group deliveries by matchId and batsman
    grouped = deliveries_df.groupby(["matchId", "batsman"], as_index=False).agg(
        batting_team=("batting_team", "first"),
        opponent=("bowling_team", "first"),
        runs=("batsman_runs", "sum"),
        balls_faced=("is_ball_faced", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
    )

    # Rename batsman to player and matchId to match_id
    grouped.rename(columns={"matchId": "match_id", "batsman": "player"}, inplace=True)

    print("[4/6] Tracking dismissals...")
    # A batter is dismissed if their name is in player_dismissed (excluding 'retired hurt')
    valid_dismissals = deliveries_df[
        (deliveries_df["player_dismissed"].notna())
        & (deliveries_df["player_dismissed"] != "")
        & (deliveries_df["player_dismissed"] != "nan")
        & (deliveries_df["dismissal_kind"] != "retired hurt")
    ]
    dismissed_pairs = set(
        zip(valid_dismissals["matchId"], valid_dismissals["player_dismissed"])
    )

    grouped["dismissed"] = [
        (mid, p) in dismissed_pairs
        for mid, p in zip(grouped["match_id"], grouped["player"])
    ]

    # Calculate strike rate: (runs / balls_faced) * 100
    grouped["strike_rate"] = np.where(
        grouped["balls_faced"] > 0,
        np.round((grouped["runs"] / grouped["balls_faced"]) * 100.0, 2),
        0.0,
    )

    print("[5/6] Merging match metadata and sorting by date...")
    # Select metadata from matches
    match_meta = matches_df[
        ["matchId", "parsed_date", "standardized_season", "standardized_venue"]
    ].copy()
    match_meta.rename(
        columns={
            "matchId": "match_id",
            "parsed_date": "date",
            "standardized_season": "season",
            "standardized_venue": "venue",
        },
        inplace=True,
    )

    # Merge match metadata with batter stats
    player_match_df = grouped.merge(match_meta, on="match_id", how="inner")

    # Sort strictly by date, then match_id, then player for consistency
    player_match_df.sort_values(
        by=["date", "match_id", "player"], ascending=[True, True, True], inplace=True
    )
    player_match_df.reset_index(drop=True, inplace=True)

    # Reformat date to standard YYYY-MM-DD string
    player_match_df["date"] = player_match_df["date"].dt.strftime("%Y-%m-%d")

    # Arrange columns in requested order
    column_order = [
        "player",
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
    player_match_df = player_match_df[column_order]

    # Verification: check for duplicate (player, match_id)
    duplicates = player_match_df.duplicated(subset=["player", "match_id"]).sum()
    print(f"Duplicate (player, match_id) check: {duplicates} duplicates found.")
    assert duplicates == 0, "Error: Duplicate (player, match_id) rows detected!"

    print(f"[6/6] Saving dataset to {output_csv}...")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    player_match_df.to_csv(output_csv, index=False)
    print(f"Successfully saved {len(player_match_df)} rows to {output_csv}.")

    return player_match_df


if __name__ == "__main__":
    df = build_player_match_dataset()
    print("\n--- DATASET SUMMARY ---")
    print(f"Shape: {df.shape}")
    print(f"Number of unique players: {df['player'].nunique()}")
    print(f"Seasons covered: {sorted(df['season'].unique())}")
    print("\nFirst 5 rows:")
    print(df.head(5).to_string())
