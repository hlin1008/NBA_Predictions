import pandas as pd

def filter_game_data(games: pd.DataFrame) -> pd.DataFrame:
    """Filter raw game data for regular season matchups and remove redundancies."""
    
    # Isolate regular season games and remove duplicates
    games = games[games["season_type"] == "Regular Season"]
    games = games.drop_duplicates(subset=["game_id"])
    
    # Drop irrelevant metadata (Note: game_date is preserved for feature engineering)
    columns_to_drop = [
        "season_id", "team_name_home", "matchup_home", "min", 
        "plus_minus_home", "video_available_home", "team_name_away", 
        "matchup_away", "plus_minus_away", "video_available_away", "wl_away"
    ]
    
    return games.drop(columns=columns_to_drop, errors='ignore')