# Courtside Inference: NBA Game Predictor

## 🚀 Running the Machine Learning Pipeline

### Prerequisites
1. **The Database:** You **must** have the full `nba.sqlite` file downloaded from Kaggle placed in the root directory. The empty shell file will crash the code.
2. **Virtual Environment:** Ensure you are running a virtual environment to avoid package conflicts.
3. **Dependencies:** Install the required ML libraries:
   `pip install pandas scikit-learn lightgbm`
   *(Mac users: If LightGBM throws an OpenMP error, run `brew install libomp`)*

### Executing the Code
Run the driver script from the terminal:
`python main.py`

This will automatically:
1. Extract games from 2010 onwards.
2. Run Feature Engineering (calculating rolling averages, handling missing data, and stripping post-game stats to prevent data leakage).
3. Train and evaluate the Baseline L1 Logistic Regression (LASSO).
4. Train and evaluate the Advanced LightGBM Classifier.
5. Output Accuracy, Log-Loss, and the top features selected by the baseline.

---

## Dataset Breakdown

### common_player_info
    Contains player's common info(birthdays, college info, etc.)
    Irrelevant to our analysis
### draft_combine_states
    Contains player's stats when first drafted, and their body metrics, basic agility tests, etc. 
    Irrelevant to our analysis.
### draft_history
    Contains each year's draft history data.
    Irrelevant to our analysis.
### game
    Contains game data of each nba game, includes teams involved, who's home court, etc. We will use lots of data from this set.
### game_info
    Basically contains the same data as 'game'.
    Irrelevant to our analysis
### game_summary
    Contains data regarding broadcasting information.
    Irrelevant to our analysis.
### inactive_players
    Contains data of all inactive players in games. We will be using this dataset since much of win/loss factors from players playing/not playing certain games.
### line_scores
    Contains games' points data by quarters. Will be used.
### officials
    Contains games' referre information. Will be used.
### other_stats
    Contains non-major stats by games. Will be used.
### play_by_play
    Contains play by play information of each game. 
    Will not be used since we are focusing on analyzing data and predicting wins before a game is played.
### player
    Contains names of players and whether they are currently playing or not.
    Irrelevant to our analysis.
### team
    Contain names, city, state, date founded information of teams.
    Irrelevant to our analysis.
### team_details
    Contains more information about teams(facebook, instagram, twitter).
    Irrelevant to our analysis.
### team_history
    Contains information about whether team is still active. 
    Irrelevant to our analysis.
### team_info_common
    Simply an empty data sheet. 

Summary: only the game, inactive_players, officials, other_stats will be used to our analysis.