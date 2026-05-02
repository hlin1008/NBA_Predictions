import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract rolling averages, fatigue metrics, and normalize data per season."""
    print("Starting Feature Engineering...")
    
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(by='game_date').reset_index(drop=True)

    # Fatigue Metric: Back-to-Back Games
    if 'game_date' in df.columns and 'team_id_home' in df.columns:
        df['days_rest_home'] = df.groupby('team_id_home')['game_date'].diff().dt.days
        df['home_back_to_back'] = (df['days_rest_home'] <= 1).astype(int)
        df['days_rest_home'] = df['days_rest_home'].fillna(5)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Temporal Normalization: Z-score scaling by season
    if 'game_date' in df.columns:
        df['year'] = df['game_date'].dt.year
        features_to_scale = [col for col in numeric_cols if col not in ['target', 'game_date', 'year', 'team_id_home', 'wl_home', 'home_back_to_back']]
        
        for col in features_to_scale:
            if col in df.columns:
                df[col] = df.groupby('year')[col].transform(lambda x: (x - x.mean()) / x.std())
        df[features_to_scale] = df[features_to_scale].fillna(0)

    # Define binary target
    if 'wl_home' in df.columns:
        df['target_win'] = df['wl_home'].apply(lambda x: 1 if x == 'W' else 0)

    # Strip leakage metrics and temporal markers used during engineering
    leakage_cols = [col for col in df.columns if any(sub in col.lower() for sub in ['pts', 'fgm', 'fga', 'reb', 'ast', 'tov', 'stl', 'blk'])]
    df = df.drop(columns=leakage_cols + ['game_date', 'year'], errors='ignore')

    print("Feature Engineering Complete.")
    return df