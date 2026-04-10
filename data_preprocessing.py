import sqlite3 as sql
import pandas as pd


def filter_game_data(games: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: games(pd.DataFrame)
    Output: pd.DataFrame
    
    Filter the game data to only include the columns we need
    '''
    games = regular_season_games(games)
    games = remove_duplicate_games(games)
    games = remove_irrelevant_data(games)
    return games


def regular_season_games(games: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: games(pd.DataFrame)
    Output: pd.DataFrame
    
    Remove all star games from the data
    ''' 
    non_all_star_idx = games["season_type"] == "Regular Season"
    games = games[non_all_star_idx]
    return games


def remove_duplicate_games(games: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: games(pd.DataFrame)
    Output: pd.DataFrame
    
    Remove duplicate games from the data
    '''
    games = games.drop_duplicates(subset=["game_id"])
    return games


def remove_irrelevant_data(games: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: games(pd.DataFrame)
    Output: pd.DataFrame
    
    Remove irrelevant data from the data
    All we need from this datasheet is essentially the date, 
    the teams, who was home court, and who won. 
    '''
    games = games.drop(columns=["season_id",  
    "team_name_home", "game_date", "matchup_home", "min", 
    "plus_minus_home", "video_available_home",  
    "team_name_away", "matchup_away", 
    "plus_minus_away", "video_available_away", "wl_away"])
    return games


