import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

def generate_standalone_graph():
    print("Connecting to database...")
    # Connect directly to your local database
    conn = sqlite3.connect('nba.sqlite')
    
    # Pull only the date and the home team's 3-point attempts
    query = """
    SELECT game_date, fg3a_home 
    FROM game 
    WHERE game_date >= '2010-01-01' AND fg3a_home IS NOT NULL
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        print(f"Successfully pulled {len(df)} games from the database.")
        
        # Convert to datetime and extract the year
        df['game_date'] = pd.to_datetime(df['game_date'])
        df['year'] = df['game_date'].dt.year
        
        # Calculate the average 3-point attempts per year
        yearly_stat = df.groupby('year')['fg3a_home'].mean()
        
        # Draw the graph
        plt.figure(figsize=(8, 5))
        plt.plot(yearly_stat.index, yearly_stat.values, marker='o', color='orange', linewidth=2)
        plt.title('The Basketball Evolution: Offensive Trends per Year')
        plt.xlabel('Year')
        plt.ylabel('Average 3-Point Attempts (Home)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('temporal_evolution.png')
        
        print("\nSUCCESS: 'temporal_evolution.png' has been generated and saved to your folder!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    generate_standalone_graph()