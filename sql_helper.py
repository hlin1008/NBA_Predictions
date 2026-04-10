import sqlite3 as sql
import pandas as pd

# CONNECT to DATABASE
conn = sql.connect("nba.sqlite")
cursor = conn.cursor()

def get_all_games() -> pd.DataFrame:
    '''
    Input: None
    Output: DataFrame of all games
    
    Get all games from the database
    '''
    query = "SELECT * FROM game"
    df = pd.read_sql_query(query, conn)
    return df

def get_games_by_season(season_id: int) -> pd.DataFrame:
    '''
    Input: cursor(sqlite3.cursor), season_id(int)
    Output: DataFrame of games
    
    Get all games for a given season
    '''
    query = create_game_query(season_id)
    df = pd.read_sql_query(query, conn)
    return df

def get_games_after_year(year: int) -> pd.DataFrame:
    '''
    Input: year(int)
    Output: DataFrame of games
    
    Get all games after a given year
    '''
    query = f"SELECT * FROM game WHERE game_date > '{year}-01-01'"
    df = pd.read_sql_query(query, conn)
    return df

def create_game_query(season_id: int) -> str:
    '''
    Input: season_id(int)
    Output: query(str)
    
    Helper function to create a query to get all games for a given season
    '''
    return f"SELECT * FROM game WHERE season_id = {season_id}"