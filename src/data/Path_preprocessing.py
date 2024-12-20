import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
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
    print("Number of different paths played:", len(df))    
    print("Mean number of time a path is played:", df['count'].mean())
    print("Median number of played path:", df['count'].median())

    # Get the value counts of the 'count' column (how often each frequency occurs)
    path_count_values = df['count'].value_counts()

    # Print the counts of paths that were played only once
    print(f"Paths that were played only once: {path_count_values.get(1, 0)}")


def process_unfinished(filename):
    """
    Opens the unfinished paths file and split it between played an untried games
    """

    # Read tsv file
    columns = ['session_id', 'timestamp', 'duration', 'path', 'target', 'type']

    unfinished_paths = pd.read_csv(filename, sep='\t', skiprows=17, header=None, names=columns)

    # Add number of pages visited for each game
    unfinished_paths['num_pages_visited'] = unfinished_paths['path'].apply(lambda x: len(x.split(';')) if pd.notna(x) else 0)


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


###verified Works

def shorten_paths_using_links(df_finished_cut, df_links):
    # Function to shorten a path based on the source and target link
    def shorten_path(path, links):
        for _, link in links.iterrows():
            # Use 'Source' and 'Target' column names
            source = link['Source']
            target = link['Target']
            
            if source in path and target in path:
                # Find indices of source and target
                source_idx = path.index(source)
                target_idx = path.index(target)
                
                # Ensure source appears before target in the path
                if source_idx < target_idx:
                    # Shorten the path: keep everything before source and after target
                    path = path[:source_idx + 1] + path[target_idx:]
        return path

    # Iterate over each row in df_finished_cut and modify the 'path_list' column
    df_finished_cut['path_list'] = df_finished_cut['path_list'].apply(
        lambda path: shorten_path(path, df_links)
    )
    
    return df_finished_cut


###Verified, works

def recalculate_num_pages_visited(df):
    """
    Recalculates the 'num_pages_visited' column by counting the number of elements in the 'path_list' column.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the 'path_list' and 'num_pages_visited' columns.
        
    Returns:
        pd.DataFrame: Updated DataFrame with recalculated 'num_pages_visited'.
    """
    # Ensure path_list exists and contains lists
    if "path_list" in df.columns:
        # Recalculate num_pages_visited
        df["num_pages_visited"] = df["path_list"].apply(len)
    else:
        raise KeyError("The column 'path_list' is not found in the DataFrame.")
    
    return df


def average_time_per_page(df):
    """
    Calculates the mean time spent on a page by players based on
    every game duration and total pages visited
    """
    total_duration = df['duration'].sum()
    total_pages = df['num_pages_visited'].sum()
    average_time = total_duration / total_pages

    return average_time

def redefine_duration(df, avg):
    """
    Correct the duration of every games in the cut dataset based on the mean time spent
    on a page
    """
    df['duration'] = df['num_pages_visited'] * avg
    return df


def compare_statistics(Original, New):
    """
    Compare statistics on wikispeedia metrics between two dataframes.
    Prints the statistics of each dataframe and the differences between them.
    """
    
    def print_stats(df, label):
        print(f"Statistics for {label} Dataset:")
        print("Mean length of paths:", df['num_pages_visited'].mean())
        print("Mean Game duration:", df['duration'].mean(), "seconds")
        print("Duration of longest game:", df['duration'].max(), "seconds")
        print("\n")
    
    # Print statistics for each dataframe
    print_stats(Original, "Original")
    print_stats(New, "New")

    # Calculate and print differences between df1 and df2
    print("Differences between Original and New datsets:")
    print("Difference in mean length of paths:", Original['num_pages_visited'].mean() - New['num_pages_visited'].mean())
    print("Difference in median path length:", Original['num_pages_visited'].median() - New['num_pages_visited'].median())
    print("Difference in mean game duration:", Original['duration'].mean() - New['duration'].mean(), "seconds")
    print("Difference in duration of longest game:", Original['duration'].max() - New['duration'].max(), "seconds")

def compare_statistics_html(original, new, output_file="comparison_statistics.html"):
    """
    Compare statistics between two dataframes and save the result as a dark-themed HTML table.
    
    Parameters:
    original (DataFrame): The original DataFrame.
    new (DataFrame): The modified DataFrame.
    output_file (str): The name of the HTML file to save.
    """
    # Compute statistics for both DataFrames
    stats = {
        "Metric": [
            "Mean length of paths", 
            "Mean Game duration", 
            "Duration of longest game"
        ],
        "Original": [
            original['num_pages_visited'].mean(), 
            original['duration'].mean(), 
            original['duration'].max()
        ],
        "New": [
            new['num_pages_visited'].mean(), 
            new['duration'].mean(), 
            new['duration'].max()
        ]
    }

    # Add differences to the stats dictionary
    stats["Difference"] = [
        stats["Original"][i] - stats["New"][i] for i in range(len(stats["Metric"]))
    ]

    # Create a DataFrame for statistics
    stats_df = pd.DataFrame(stats)

    # Convert to an HTML table with a dark theme
    html_table = stats_df.to_html(index=False, classes="table table-bordered", justify="center")
    
    # Dark-themed styling
    dark_theme = """
    <style>
        body {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        h2 {
            color: #ffffff;
            text-align: center;
        }
        .table {
            width: 80%;
            margin: auto;
            border-collapse: collapse;
            color: #ffffff;
        }
        .table-bordered {
            border: 1px solid #555555;
        }
        .table-bordered th, .table-bordered td {
            border: 1px solid #555555;
            padding: 10px;
            text-align: center;
        }
        .table th {
            background-color: #333333;
        }
        .table tr:nth-child(even) {
            background-color: #2a2a2a;
        }
        .table tr:nth-child(odd) {
            background-color: #1e1e1e;
        }
    </style>
    """

    # Save the HTML table to a file
    with open(output_file, "w") as f:
        f.write("<html><head>")
        f.write(dark_theme)
        f.write("</head><body>")
        f.write("<h2>Comparison of Original and New Statistics</h2>")
        f.write(html_table)
        f.write("</body></html>")

    print(f"Comparison statistics have been saved to {output_file} with a dark background.")

