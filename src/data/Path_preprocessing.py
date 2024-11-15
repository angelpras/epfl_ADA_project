import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import math
from collections import Counter


def process_finished(filename):
    """
    Opens the finished paths file and arrange data
    """

    columns = ['session_id', 'timestamp', 'duration', 'path', 'rating']

    # Open tsv file
    finished_paths = pd.read_csv(filename, sep='\t', skiprows=15, header=None, names=columns)

    # Filter out sequences of "Wikipedia_Text....;<"
    finished_paths['path'] = finished_paths['path'].str.replace("Wikipedia_Text_of_the_GNU_Free_Documentation_License;<", "", regex=False)

    # Add the number of pages visited for each game
    finished_paths['num_pages_visited'] = finished_paths['path'].apply(lambda x: len(x.split(';')) if pd.notna(x) else 0)


    return finished_paths

def statistics(df):
    """"
    Print statistics on wikispeedia metrics
    """

    print("Mean length of paths:",df['num_pages_visited'].mean())
    print("Median path length:", df['num_pages_visited'].median())
    print("Mean Game duration:", df['duration'].mean(), "seconds")
    print("Median Game duration:", df['duration'].median(), "seconds")
    print("Maximum pages visited:", df['num_pages_visited'].idxmax())
    print("Duration of longest game:", df['duration'].idxmax(), "seconds")    

def plot_num_pages(df):
    """
    Plot the distribution of games based on the number of pages visited
    """

    # Define the bins for the ranges of pages visited
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 100, 200, 500]

    # Use pd.cut to categorize the 'num_pages_visited' into these bins
    df['page_range'] = pd.cut(df['num_pages_visited'], bins)

    # Count the number of games in each bin
    page_range_counts = df['page_range'].value_counts().sort_index()

    # Plot the results
    plt.figure(figsize=(10, 6))
    page_range_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Games by Page Range Visited')
    plt.xlabel('Number of Pages Visited (Range)')
    plt.ylabel('Number of Games')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

def most_visited(df):
    """
    Creates a df of visited pages and displays the 10 most visited pages
    """

    # Split the 'path' column into lists of pages
    df['path_list'] = df['path'].str.split(';')

    # Flatten the list of pages and remove any NaN or empty values
    all_pages = df['path_list'].dropna().explode()

    # Filter out the '<' character
    filtered_pages = [page for page in all_pages if page != '<']

    # Count the frequency of each page
    page_counts = Counter(filtered_pages)

    # Get the 10 most common pages
    top_10_pages = page_counts.most_common(10)

    # Print the top 10 most visited pages
    for page, count in top_10_pages:
        print(f"Page: {page}, Visits: {count}")
    
    return page_counts

def number_games(df):
    """
    Creates a df of paths played and displays the 10 most played paths
    """
    # Split the 'path' column into a list of pages
    df['path_list'] = df['path'].dropna().str.split(';')

    # Extract the starting and ending pages of each path
    df['start_page'] = df['path_list'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['end_page'] = df['path_list'].apply(lambda x: x[-1] if len(x) > 0 else None)

    # Group by 'start_page' and 'end_page' and count the number of occurrences
    path_counts = df.groupby(['start_page', 'end_page']).size().reset_index(name='count')

    # Sort the DataFrame by 'count' in descending order and get the top 10 paths
    top_10_paths = path_counts.sort_values(by='count', ascending=False).head(10)

    # Print the top 10 paths
    print("Top 10 most frequent paths:")
    print(top_10_paths)

    return path_counts

def stats_on_games(df):
    """
    Print statistics on the number of time same games are played
    """
        
    print("Mean number of time a path is played:", df['count'].mean())
    print("Median number of played path:", df['count'].median())

def process_unfinished(filename):
    """
    Opens the unfinished paths file and split it between played an untried games
    """

    # Read tsv file
    columns = ['session_id', 'timestamp', 'duration', 'path', 'target', 'type']

    unfinished_paths = pd.read_csv(filename, sep='\t', skiprows=17, header=None, names=columns)

    #Remove played paths ending on "Wikipidia GNU..."
    unfinished_paths['path_list'] = unfinished_paths['path'].str.split(';')
    unfinished_paths_filtered = unfinished_paths[unfinished_paths['path_list'].apply(lambda x: x[-1] != "Wikipedia_Text_of_the_GNU_Free_Documentation_License")]

    #Create subset of non played games (1 page visited)
    not_played_unfinished = unfinished_paths_filtered[unfinished_paths['num_pages_visited'] == 1]

    #Create a subset of played games (more than 1 page visited)
    played_unfinished = unfinished_paths_filtered[unfinished_paths['num_pages_visited'] > 1]

    return played_unfinished, not_played_unfinished

def stats_unfinished(df_played, df_unplayed):
    """
    Displays statistics on the unfinished paths data
    """

    print(f"There are : {len(df_played) + len(df_unplayed)} unfinished games")
    print(f"There are : {len(df_played)} failed games and {len(df_unplayed)} not attempted games")
    print("Mean length of paths:",df_played['num_pages_visited'].mean())
    print("Median path length:", df_played['num_pages_visited'].median())
    print("Mean Game duration:", df_played['duration'].mean(), "seconds")
    print("Median Game duration:", df_played['duration'].median(), "seconds")
    print("Maximum pages visited:", df_played['num_pages_visited'].idxmax())
    print("Duration of longest game:", df_played['duration'].idxmax(), "seconds") 
