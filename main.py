# LIBARY IMPORTS
import pandas as pd
import sqlite3 as sql

from pandas.io.sql import import_optional_dependency

# HELPER FUNCTIONS
import sql_helper as sql_helper
import data_preprocessing as dp

# Retrieve DATA from DATABASE
all_games = sql_helper.get_all_games()
games_after_2010 = sql_helper.get_games_after_year(2010)
games_filtered = dp.filter_game_data(games_after_2010)