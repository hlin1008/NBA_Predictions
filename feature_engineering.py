import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    EC503 Concept: Feature Extraction & Handling Time-Series Data
    We need to transform raw box scores into predictive features 
    that represent team momentum and fatigue.
    """
    print("Starting Feature Engineering...")
    
    # 1. Sort chronologically to prevent Data Leakage! 
    # We cannot use future games to predict past games.
    if 'game_date' in df.columns:
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(by='game_date').reset_index(drop=True)

    # 2. Simulate Fatigue: The "Back-to-Back" feature
    # If a team played yesterday, they are likely fatigued.
    # Note: This is a simplified proxy. We will calculate the days since the last game.
    # (Assuming we have a 'team_id' and 'game_date' in the final merged data)
    
    # Placeholder logic (will adjust once we see Hanks's exact columns):
    # df['days_rest_home'] = df.groupby('team_id_home')['game_date'].diff().dt.days
    # df['home_back_to_back'] = (df['days_rest_home'] <= 1).astype(int)

    # 3. Handle Missing Values
    # Traditional ML methods (like Logistic Regression) cannot handle NaNs natively.
    # We will fill missing statistical columns with the median to avoid skewing our weights.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # 4. Target Variable r
    # Ensure our target variable (Win/Loss) is strictly binary (1 or 0)
    if 'wl_home' in df.columns:
        df['target_win'] = df['wl_home'].apply(lambda x: 1 if x == 'W' else 0)

    print("Feature Engineering Complete.")
    
    # 5. Prevent Data Leakage
    # We must drop post-game stats. If the model knows how many points 
    # were scored, it's not predicting; it's just reading the score!
    leakage_cols = [col for col in df.columns if 'pts' in col.lower() or 'fgm' in col.lower() or 'fga' in col.lower() or 'reb' in col.lower() or 'ast' in col.lower() or 'tov' in col.lower() or 'stl' in col.lower() or 'blk' in col.lower()]
    df = df.drop(columns=leakage_cols, errors='ignore')

    return df