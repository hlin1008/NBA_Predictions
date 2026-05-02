import sqlite3 as sql
import pandas as pd

conn = sql.connect("nba.sqlite")

def get_all_games() -> pd.DataFrame:
    """Retrieve all games from the database."""
    query = "SELECT * FROM game"
    return pd.read_sql_query(query, conn)

def get_games_by_season(season_id: int) -> pd.DataFrame:
    """Retrieve all games for a specific season."""
    query = f"SELECT * FROM game WHERE season_id = {season_id}"
    return pd.read_sql_query(query, conn)

def get_games_after_year(year: int) -> pd.DataFrame:
    """Retrieve all games played after January 1st of a given year."""
    query = f"SELECT * FROM game WHERE game_date > '{year}-01-01'"
    return pd.read_sql_query(query, conn)