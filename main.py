# LIBRARY IMPORTS
import pandas as pd
import sqlite3 as sql

# HELPER IMPORTS
import sql_helper as sql_helper
import data_preprocessing as dp
import feature_engineering as fe
import model_training as mt 

print("1. Extracting games after 2010 from database...")
games_after_2010 = sql_helper.get_games_after_year(2010)

print("2. Preprocessing data (filtering duplicates & All-Star games)...")
games_filtered = dp.filter_game_data(games_after_2010)

print("3. Running Feature Engineering...")
games_engineered = fe.engineer_features(games_filtered)

print(f"Data successfully prepared! Final dataset shape: {games_engineered.shape}")

# Run the machine learning pipeline
mt.run_models(games_engineered)