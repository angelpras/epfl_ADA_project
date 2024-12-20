import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import dot_score

def visualize_graph(G):
    """
    Displays the connected graph G  
    """
    # Compute the nodes positions with layout 
    pos = nx.spring_layout(G, k=0.15, iterations=50, weight=None)

    # Optionnal, adjust the size of the node
    node_sizes = [G.degree(node) * 1 for node in G.nodes()]

    # Draw the graph
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, with_labels=False, node_size=node_sizes, node_color='skyblue',
            font_size=10, font_weight='bold', edge_color='gray', arrows=True)

    plt.title("Connected Graph of Articles and Links with Embeddings")
    plt.axis('off') 
    #plt.savefig("graph.png", dpi=1000)  
    plt.show()

def create_subset_unconnected_nodes(G, subset_size=350, seed=1):
    random.seed(seed)
    # New way of taking a subset of unconnected nodes
    connected_pairs = set(G.edges())
    all_nodes = list(G.nodes)
    unconnected_pairs = set()

    while len(unconnected_pairs) < subset_size**2:
        # Randomly sample two distinct nodes
        a, b = random.sample(all_nodes, 2)
        # Add the pair if it is not connected and not already in unconnected_pairs
        if (a, b) not in connected_pairs and (b, a) not in connected_pairs:
            unconnected_pairs.add((a, b))

    return unconnected_pairs

def create_node_similarity_distributions(G, unconnected_pairs):
    weight_titles_connected = [data['weight_title'] for _, _, data in G.edges(data=True)]
    weight_descriptions_connected = [data['weight_description'] for _, _, data in G.edges(data=True)]
    unconnected_similarities = []
    weight_titles_unconnected = []
    weight_descriptions_unconnected = []


    for source, target in unconnected_pairs:
        embedding_title_source = G.nodes[source]['embedding_title']
        embedding_description_source = G.nodes[source]['embedding_description']
        embedding_title_target = G.nodes[target]['embedding_title']
        embedding_description_target = G.nodes[target]['embedding_description']
        
        cosine_title = float(dot_score(embedding_title_source, embedding_title_target))
        cosine_description = float(dot_score(embedding_description_source, embedding_description_target))

        weight_titles_unconnected.append(cosine_title)
        weight_descriptions_unconnected.append(cosine_description)

        unconnected_similarities.append({
            'source': source,
            'target': target,
            'title_similarity': cosine_title,
            'description_similarity': cosine_description
        })
    
    return {
        'connected_titles': weight_titles_connected,
        'connected_descriptions': weight_descriptions_connected,
        'unconnected_titles': weight_titles_unconnected,
        'unconnected_descriptions': weight_descriptions_unconnected,
        'unconnected_pairs': unconnected_similarities
    }

def visualize_node_similarity_distributions(unconnected_pairs, G=None, similarities=None, y_max=12500):
    if similarities is None:
        if G is None:
            raise ValueError("You must provide a graph G")
        similarities = create_node_similarity_distributions(G, unconnected_pairs)

    weight_titles_connected = similarities['connected_titles']
    weight_descriptions_connected = similarities['connected_descriptions']
    weight_titles_unconnected = similarities['unconnected_titles']
    weight_descriptions_unconnected = similarities['unconnected_descriptions']

    # Plot the histograms
    all_weights_titles = weight_titles_connected + weight_titles_unconnected
    all_weights_descriptions = weight_descriptions_connected + weight_descriptions_unconnected

    x_min = min(min(all_weights_titles), min(all_weights_descriptions))
    x_max = max(max(all_weights_titles), max(all_weights_descriptions))

    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribution of Cosine Similarity for Connected and Unconnected Nodes', fontsize=18)

    axes[0, 0].hist(weight_titles_connected, bins=50, edgecolor='black') 
    axes[0, 0].set_title('Connected Nodes - Title Similarity')
    axes[0, 0].set_xlabel('Cosine Similarity')
    axes[0, 0].set_ylabel('Number of Occurrences')
    axes[0, 0].grid(axis='y', alpha=0.75)
    axes[0, 0].set_xlim(x_min, x_max)
    axes[0, 0].set_ylim(0, y_max)

    axes[0, 1].hist(weight_descriptions_connected, bins=50, edgecolor='black') 
    axes[0, 1].set_title('Connected Nodes - Description Similarity')
    axes[0, 1].set_xlabel('Cosine Similarity')
    axes[0, 1].set_ylabel('Number of Occurrences')
    axes[0, 1].grid(axis='y', alpha=0.75)
    axes[0, 1].set_xlim(x_min, x_max)
    axes[0, 1].set_ylim(0, y_max)

    axes[1, 0].hist(weight_titles_unconnected, bins=50, edgecolor='black')
    axes[1, 0].set_title('Unconnected Nodes - Title Similarity')
    axes[1, 0].set_xlabel('Cosine Similarity')
    axes[1, 0].set_ylabel('Number of Occurrences')
    axes[1, 0].grid(axis='y', alpha=0.75)
    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_ylim(0, y_max)

    axes[1, 1].hist(weight_descriptions_unconnected, bins=50, edgecolor='black')
    axes[1, 1].set_title('Unconnected Nodes - Description Similarity')
    axes[1, 1].set_xlabel('Cosine Similarity')
    axes[1, 1].set_ylabel('Number of Occurrences')
    axes[1, 1].grid(axis='y', alpha=0.75)
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(0, y_max)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def visualize_connected_vs_unconnected_cs_distribution(similarities):
    # Box plots for titles
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=[similarities['connected_titles'], similarities['unconnected_titles']])
    plt.xticks([0, 1], ['Connected Nodes - Titles', 'Unconnected Nodes - Titles'])
    plt.ylabel('Cosine Similarity')
    plt.title('Box Plot of Cosine Similarity in Titles')
    plt.show()

    # Box plots for descriptions
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=[similarities['connected_descriptions'], similarities['unconnected_descriptions']])
    plt.xticks([0, 1], ['Connected Nodes - Descriptions', 'Unconnected Nodes - Descriptions'])
    plt.ylabel('Cosine Similarity')
    plt.title('Box Plot of Cosine Similarity in Descriptions')
    plt.show()


def calculate_links_conditional_proba(similarities):

    # Define similarity bins 
    bins = np.arange(-0.4, 1.05, 0.05)

    # Calculate histograms (frequencies) for connected and unconnected nodes
    connected_descriptions_counts, _ = np.histogram(similarities['connected_descriptions'], bins=bins)
    unconnected_descriptions_counts, _ = np.histogram(similarities['unconnected_descriptions'], bins=bins)
    connected_titles_counts, _ = np.histogram(similarities['connected_titles'], bins=bins)
    unconnected_titles_counts, _ = np.histogram(similarities['unconnected_titles'], bins=bins)

    # Create a DataFrame to store the results
    df_descriptions = pd.DataFrame({
        'bin_center': bins[:-1] + 0.025,  # Center of each bin
        'connected': connected_descriptions_counts,
        'unconnected': unconnected_descriptions_counts
    })
    df_titles = pd.DataFrame({
        'bin_center': bins[:-1] + 0.025,  # Center of each bin
        'connected': connected_titles_counts,
        'unconnected': unconnected_titles_counts
    })

    # Calculate the conditional probability of a link in each bin
    df_descriptions['total'] = df_descriptions['connected'] + df_descriptions['unconnected']
    df_descriptions['p(link|similarity)'] = df_descriptions['connected'] / df_descriptions['total']
    df_titles['total'] = df_titles['connected'] + df_titles['unconnected']
    df_titles['p(link|similarity)'] = df_titles['connected'] / df_titles['total']

    # Plot the conditional probability graph versus cosine similarity
    plt.figure(figsize=(10, 6))
    plt.bar(df_descriptions['bin_center'], df_descriptions['p(link|similarity)'], width=0.05, color='skyblue', edgecolor='black')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Estimated probability of a link between two random nodes')
    plt.title('Estimated probability of a link between two random nodes according to cosine similarity distribution with articles descriptions')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(df_titles['bin_center'], df_titles['p(link|similarity)'], width=0.05, color='skyblue', edgecolor='black')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Estimated probability of a link between two random nodes')
    plt.title('Estimated probability of a link between two random nodes according to cosine similarity distribution with articles titles')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def calculate_preferential_attachment(G, unconnected_pairs):
    # Convert to undirected graph and calculate scores
    G_undirected = G.to_undirected()
    
    # Calculate scores for connected and unconnected pairs
    attachment_connected_vals = nx.preferential_attachment(G_undirected)
    attachment_connected_scores = [{'source': u, 'target': v, 'score': j} 
                                 for u, v, j in attachment_connected_vals]
    
    attachment_unconnected_vals = nx.preferential_attachment(G_undirected, ebunch=unconnected_pairs)
    attachment_unconnected_scores = [{'source': u, 'target': v, 'score': j} 
                                   for u, v, j in attachment_unconnected_vals]
    
    # Extract scores for plotting
    connected_scores = [entry['score'] for entry in attachment_connected_scores]
    unconnected_scores = [entry['score'] for entry in attachment_unconnected_scores]
    
    # Plot connected pairs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Preferential Attachment Scores for Connected Pairs | Same x-axis', fontsize=16)
    
    ax1.hist(connected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Preferential Attachment Scores for Node Pairs | Connected Pairs')
    ax1.set_xlabel('Preferential Attachment Score')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.set_xlim(0, np.max(connected_scores))
    ax1.set_ylim(1e0, 1e7)
    
    ax2.scatter(range(len(connected_scores)), connected_scores, color='blue', alpha=0.5)
    ax2.set_title('Preferential Attachment Scores for Node Pairs | Connected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Preferential Attachment Score')
    ax2.set_ylim(0, np.max(connected_scores))
    
    plt.tight_layout()
    plt.show()
    
    # Plot unconnected pairs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Preferential Attachment Scores for Unconnected Pairs', fontsize=16)
    
    ax1.hist(unconnected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Preferential Attachment Scores | Graph Subset | Unconnected Pairs')
    ax1.set_xlabel('Preferential Attachment Score')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.set_xlim(0, np.max(connected_scores))
    ax1.set_ylim(1e0, 1e7)
    
    ax2.scatter(range(len(unconnected_scores)), unconnected_scores, color='blue', alpha=0.5)
    ax2.set_title('Preferential Attachment Scores for Node Pairs | Graph Subset | Unconnected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Preferential Attachment Score')
    ax2.set_ylim(0, np.max(connected_scores))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'connected_scores': attachment_connected_scores,
        'unconnected_scores': attachment_unconnected_scores
    }

def calculate_preferential_attachment_unconnected_zoomed(G, unconnected_pairs):
    ############ Connected Nodes ############
    G_undirected = G.to_undirected()
    ############ Unconnected Nodes ############
    attachment_unconnected_scores = [score for _, _, score in nx.preferential_attachment(G_undirected, ebunch=unconnected_pairs)]

    ############ Plotting Preferential Score Frequency and Values per Node Pairs ############
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Preferential Attachment Scores for Unconnected Pairs', fontsize=16)
    ax1.hist(attachment_unconnected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Preferential Attachment Scores | Graph Subset | Unconnected Pairs')
    ax1.set_xlabel('Preferential Attachment Score')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    #ax1.set_xscale('log')
    ax1.set_ylim(1e0, 1e7)
    #ax1.set_xlim(0,np.max(attachment_connected_scores))
    #ax1.set_xlim(0,np.max(attachment_connected_scores))
    ax2.scatter(range(len(attachment_unconnected_scores)), attachment_unconnected_scores, color='blue', alpha=0.5)
    ax2.set_title('Preferential Attachment Scores for Node Pairs | Graph Subset | Unconnected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Preferential Attachment Score')
    #ax2.set_yscale('log')
    plt.tight_layout()
    plt.show()


def calculate_common_neighbors(G, unconnected_pairs, subset_size=350):
    ############ Connected Nodes ############
    G_undirected = G.to_undirected()
    edges = G_undirected.edges()
    common_neighbors_connected_counts = []
    for u, v in edges:
        common_neighbors_connected_u_v = list(nx.common_neighbors(G_undirected, u, v))  
        common_neighbors_connected_counts.append(len(common_neighbors_connected_u_v))  

    ############ Unconnected Nodes ############
    # Remove direction and find attachment scores for unconnected nodes
    unconnected_pairs = set(random.sample(list(unconnected_pairs), min(len(unconnected_pairs), subset_size**2)))
    common_neighbors_unconnected_counts = []
    for u, v in unconnected_pairs:
        common_neighbors_unconnected_u_v = list(nx.common_neighbors(G_undirected, u, v))  
        common_neighbors_unconnected_counts.append(len(common_neighbors_unconnected_u_v))  

    ############ Plotting for Common Neighbors ############

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Distribution of Common Neighbors', fontsize=16)
    ax1.hist(common_neighbors_connected_counts, bins=50, edgecolor='black')
    ax1.set_title('Distribution of Common Neighbors | Connected Pairs')
    ax1.set_xlabel('Number of Common Neighbors')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.set_xlim(0,np.max(common_neighbors_connected_counts))
    ax1.set_ylim(1e0,1e5)
    #ax1.set_xlim(0,np.max(common_neighbors_connected_counts)/3)
    ax2.hist(common_neighbors_unconnected_counts, bins=50, edgecolor='black')
    ax2.set_title('Distribution of Common Neighbors | Unconnected Pairs')
    ax2.set_xlabel('Number of Common Neighbors')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    ax2.set_ylim(1e0,1e5)
    ax2.set_xlim(0,np.max(common_neighbors_connected_counts))
    plt.tight_layout()
    plt.show()

def calculate_jaccards_coeff(G, unconnected_pairs, plot=True):
    ############ Connected Nodes ############
    G_undirected = G.to_undirected()
    jaccard_connected_vals = nx.jaccard_coefficient(G_undirected)
    jaccard_connected_scores = [{'source': u, 'target': v, 'score': j} for u, v, j in jaccard_connected_vals]

    ############ Unconnected Nodes ############
    jaccard_unconnected_vals = nx.jaccard_coefficient(G_undirected, ebunch=unconnected_pairs)
    jaccard_unconnected_scores = [{'source': u, 'target': v, 'score': j} for u, v, j in jaccard_unconnected_vals]

    if plot:
        ############ Plotting Jaccard's Coefficient Frequency and Values per Node Pairs ############
        connected_scores_only = [entry['score'] for entry in jaccard_connected_scores]
        unconnected_scores_only = [entry['score'] for entry in jaccard_unconnected_scores]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
        fig.suptitle('Jaccard\'s Coefficient for Connected Pairs', fontsize=16)
        ax1.hist(connected_scores_only, bins=50, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Jaccard\'s Coefficient for Node Pairs | Connected Pairs')
        ax1.set_xlabel('Jaccard\'s Coefficient')
        ax1.set_ylabel('Frequency')
        ax1.set_yscale('log')
        ax1.set_ylim(1e0,1e7)
        ax1.set_xlim(0,np.max(connected_scores_only))
        ax2.scatter(range(len(connected_scores_only)), connected_scores_only, color='blue', alpha=0.5)
        ax2.set_title('Jaccard\'s Coefficient for Node Pairs | Connected Pairs')
        ax2.set_xlabel('Node Pair Index')
        ax2.set_ylabel('Jaccard\'s Coefficient')
        plt.tight_layout()
        plt.show()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
        fig.suptitle('Jaccard\'s Coefficient for Unconnected Pairs', fontsize=16)
        ax1.hist(unconnected_scores_only, bins=50, color='skyblue', edgecolor='black')
        ax1.set_title('Distribution of Jaccard\'s Coefficient | Graph Subset | Unconnected Pairs')
        ax1.set_xlabel('Jaccard\'s Coefficient')
        ax1.set_ylabel('Frequency')
        ax1.set_yscale('log')
        ax1.set_ylim(1e0,1e7)
        ax1.set_xlim(0,np.max(unconnected_scores_only))
        ax2.scatter(range(len(unconnected_scores_only)), unconnected_scores_only, color='blue', alpha=0.5)
        ax2.set_title('Jaccard\'s Coefficient for Node Pairs | Graph Subset | Unconnected Pairs')
        ax2.set_xlabel('Node Pair Index')
        ax2.set_ylabel('Jaccard\'s Coefficient')
        plt.tight_layout()
        plt.show()

    return {
        'connected_scores': jaccard_connected_scores,
        'unconnected_scores': jaccard_unconnected_scores
    }

def calculate_adamic_adar(G, unconnected_pairs):
    ############ Connected Nodes ############
    G_undirected = G.to_undirected()
    adar_connected_vals = nx.adamic_adar_index(G_undirected)
    adar_connected_scores = [{'source': u, 'target': v, 'score': j} for u, v, j in adar_connected_vals]

    ############ Unconnected Nodes ############
    adar_unconnected_vals = nx.adamic_adar_index(G_undirected, ebunch=unconnected_pairs)
    adar_unconnected_scores = [{'source': u, 'target': v, 'score': j} for u, v, j in adar_unconnected_vals]


    ############ Plotting Adamic/Adar Coefficient Frequency and Values per Node Pairs ############
    connected_scores_only = [entry['score'] for entry in adar_connected_scores]
    unconnected_scores_only = [entry['score'] for entry in adar_unconnected_scores]
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Adamic/Adar Coefficient for Connected Pairs', fontsize=16)
    ax1.hist(connected_scores_only, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Adamic/Adar Coefficient for Node Pairs | Connected Pairs')
    ax1.set_xlabel('Adamic/Adar Coefficient')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.set_ylim(1e0,1e7)
    ax1.set_xlim(0,np.max(connected_scores_only))
    ax2.scatter(range(len(connected_scores_only)), connected_scores_only, color='blue', alpha=0.5)
    ax2.set_title('Adamic/Adar Coefficient for Node Pairs | Connected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Adamic/Adar Coefficient')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Adamic/Adar Coefficient for Unconnected Pairs', fontsize=16)
    ax1.hist(unconnected_scores_only, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Adamic/Adar Coefficient | Graph Subset | Unconnected Pairs')
    ax1.set_xlabel('Adamic/Adar Coefficient')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.set_ylim(1e0,1e7)
    ax1.set_xlim(0,np.max(unconnected_scores_only))
    ax2.scatter(range(len(unconnected_scores_only)), unconnected_scores_only, color='blue', alpha=0.5)
    ax2.set_title('Adamic/Adar Coefficient for Node Pairs | Graph Subset | Unconnected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Adamic/Adar Coefficient')
    ax2.set_ylim(0,np.max(unconnected_scores_only))
    plt.tight_layout()
    plt.show()

    return {
        'connected_scores': adar_connected_scores,
        'unconnected_scores': adar_unconnected_scores
    }

def calculate_conditional_probability(scores, metric_name, bins=None):
    """
    Calculates and visualizes the conditional probability of a link existing between nodes
    based on their similarity scores (Adamic-Adar or Jaccard).
    
    Args:
        scores: Dictionary containing similarity scores for both connected and unconnected pairs
               in format {'metric_connected_scores': [{'score': float}, ...],
                         'metric_unconnected_scores': [{'score': float}, ...]}
        metric_name: String indicating the metric being used ('Adamic-Adar' or 'Jaccard')
        bins: Optional numpy array of bin edges. If None, will be automatically determined
              based on the data range
    
    Returns:
        tuple: (DataFrame with probability calculations, Figure object)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    connected_scores = np.array([entry['score'] for entry in scores['connected_scores']])
    unconnected_scores = np.array([entry['score'] for entry in scores['unconnected_scores']])
    
    # Determine appropriate bins if not provided
    if bins is None:
        min_score = min(np.min(connected_scores), np.min(unconnected_scores))
        max_score = max(np.max(connected_scores), np.max(unconnected_scores))
        bins = np.linspace(min_score, max_score, 30)  # 30 bins spanning the data range
    
    # Calculate histograms
    connected_counts, bin_edges = np.histogram(connected_scores, bins=bins)
    unconnected_counts, _ = np.histogram(unconnected_scores, bins=bins)
    
    # Normalize the counts
    connected_normalized = connected_counts / np.sum(connected_counts)
    unconnected_normalized = unconnected_counts / np.sum(unconnected_counts)
    
    # Create DataFrame
    df = pd.DataFrame({
        'bin_center': (bin_edges[1:] + bin_edges[:-1]) / 2,
        'connected': connected_counts,
        'unconnected': unconnected_counts,
        'connected_normalized': connected_normalized,
        'unconnected_normalized': unconnected_normalized
    })
    
    # Calculate conditional probability
    df['total_normalized'] = df['connected_normalized'] + df['unconnected_normalized']
    df['p(link|similarity)'] = np.where(
        df['total_normalized'] > 0,
        df['connected_normalized'] / df['total_normalized'],
        0
    )
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.bar(df['bin_center'], 
            df['p(link|similarity)'], 
            width=np.diff(bins)[0] * 0.9,  # slightly smaller than bin width
            color='skyblue', 
            edgecolor='black')
    
    plt.xlabel(f'{metric_name} Score')
    plt.ylabel('P(link|similarity)')
    plt.title(f'Conditional Probability of Link Given {metric_name} Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    return df, plt.gcf()

def analyze_graph_statistics(G):
    """
    In this function, some characteristics of the graph are computed 
    """
    # Number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    #average degree
    degrees = [deg for _,deg in G.degree()]
    average_deg=np.mean(degrees)

    # Degree distribution
    plt.figure()
    plt.hist(degrees,bins=40,log=True,edgecolor='black')
    plt.xlabel("Nodes Degrees")
    plt.ylabel("Occurences")
    plt.title("Degree Distribution")
    plt.show()
    # Network density
    density = nx.density(G)

    # Clustering coefficient
    clustering_coeff = nx.average_clustering(G)
    #Average shortest path length
    avg_path_length = nx.average_shortest_path_length(G)

    #Print results
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average degree: {average_deg:.2f}")
    print(f"Network density: {density:.4f}")
    print(f"Clustering coefficient: {clustering_coeff:.4f}")
    print(f"Average Shortest path: {avg_path_length:.4f}")
def Visualization_categories_distribution_premodel(df_links_,df_categories):
    df_links_source=df_links_.copy()
    df_links_source = df_links_source.merge(df_categories[['Article', 'Category_Level_1']], 
                            left_on='Source', right_on='Article', how='left')

    # Rename Source categories
    df_links_source = df_links_source.rename(columns={
        'Category_Level_1': 'Source_Category_1',
    })
    df_links_target=df_links_.copy()
    # Merge categories for the Target nodes
    df_links_target = df_links_target.merge(df_categories[['Article', 'Category_Level_1']], 
                            left_on='Target', right_on='Article', how='left')

    # Rename Target categories
    df_links_target = df_links_target.rename(columns={
        'Category_Level_1': 'Target_Category_1',
    })

    # Combine all categories (Source and Target) into one column
    source_columns = ['Source_Category_1']
    target_columns = ['Target_Category_1']

    source_categories = pd.concat([df_links_source[col] for col in source_columns], ignore_index=True)
    target_categories = pd.concat([df_links_target[col] for col in target_columns], ignore_index=True)
    all_categories = pd.concat([source_categories,target_categories])

    # Count the frequency of categories 
    category_counts = all_categories.value_counts().reset_index()
    category_counts.columns = ['Category', 'Frequency']
    source_category_counts = source_categories.value_counts().reset_index()[:10]
    source_category_counts.columns = ['Category', 'Frequency']
    target_category_counts = target_categories.value_counts().reset_index()[:10]
    target_category_counts.columns = ['Category', 'Frequency']

    # Drop NaN values
    #all_categories = all_categories.dropna()

    # Create an interactive bar chart using Plotly
    fig = px.bar(
        category_counts,
        x='Category',
        y='Frequency',
        title='Distribution of Categories of Graph Edges',
        labels={'Frequency': 'Count', 'Category': 'Category'},
        hover_data={'Category': True, 'Frequency': True},
        text='Frequency',
        color='Frequency',
        color_continuous_scale=px.colors.sequential.Sunset
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        xaxis=dict(tickangle=45, title='Category'),
        yaxis=dict(title='Frequency'),
        template='plotly_dark',
        title_font_size=20,
        title_x=0.5,
        margin=dict(l=50, r=50, t=80, b=150),
        height=600,
        width=1000
    )
    fig.show("png")

    # Plot Source Category Distribution using Plotly
    fig_source = px.bar(
        source_category_counts,
        x='Category',
        y='Frequency',
        title='Distribution of Categories for Source Nodes',
        labels={'Frequency': 'Count', 'Category': 'Category'},
        hover_data={'Category': True, 'Frequency': True},
        text='Frequency',
        color='Frequency',
        color_continuous_scale=px.colors.sequential.Sunset
    )

    fig_source.update_traces(texttemplate='%{text}', textposition='outside')
    fig_source.update_layout(
        xaxis=dict(tickangle=45, title='Category'),
        yaxis=dict(title='Frequency'),
        template='plotly_dark',
        title_font_size=20,
        title_x=0.5,
        margin=dict(l=50, r=50, t=80, b=150),
        height=600,
        width=1000
    )
    fig_source.show("png")

    # Plot Target Category Distribution using Plotly
    fig_target = px.bar(
        target_category_counts,
        x='Category',
        y='Frequency',
        title='Distribution of Categories for Target Nodes',
        labels={'Frequency': 'Count', 'Category': 'Category'},
        hover_data={'Category': True, 'Frequency': True},
        text='Frequency',
        color='Frequency',
        color_continuous_scale=px.colors.sequential.Sunset
    )

    fig_target.update_traces(texttemplate='%{text}', textposition='outside')
    fig_target.update_layout(
        xaxis=dict(tickangle=45, title='Category'),
        yaxis=dict(title='Frequency'),
        template='plotly_dark',
        title_font_size=20,
        title_x=0.5,
        margin=dict(l=50, r=50, t=80, b=150),
        height=600,
        width=1000
    )
    fig_target.show("png")


def Visualization_post_model(df_links):
    # Create Graph from Predicted Links
    G = nx.from_pandas_edgelist(df_links, source="Source", target="Target", create_using=nx.DiGraph())

    # 1. Frequency Analysis
    source_counts = df_links["Source"].value_counts()
    target_counts = df_links["Target"].value_counts()

    fig_source = px.bar(
        x=source_counts.head(10).index,
        y=source_counts.head(10).values,
        title="Top 10 Sources",
        labels={'x': 'Source', 'y': 'Frequency'},
        text=source_counts.head(10).values,  # Add text for the bar values
        color=source_counts.head(10).values,
        color_continuous_scale=px.colors.sequential.Sunset
    )

    # Adjust text position and layout
    fig_source.update_traces(
        texttemplate='%{text}',  # Show the text
        textposition='outside'  # Move text outside the bar
    )
    fig_source.update_layout(
        xaxis=dict(tickangle=45),
        margin=dict(t=80, b=50, l=50, r=50),  # Adjust top margin to prevent text clipping
        yaxis=dict(range=[0, source_counts.head(10).values.max() + 1])  # Add space above bars
    )

    fig_source.show("png")

    # Interactive Plot for Top 10 Targets
    fig_target = px.bar(
        x=target_counts.head(10).index,
        y=target_counts.head(10).values,
        title="Top 10 Targets",
        labels={'x': 'Target', 'y': 'Frequency'},
        text=target_counts.head(10).values,
        color=target_counts.head(10).values,
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig_target.update_traces(texttemplate='%{text}', textposition='outside')
    fig_target.update_layout(xaxis=dict(tickangle=45))
    fig_target.show("png")

    # 2. Degree Distribution
    print("Degree Distribution")
    degrees = [deg for _, deg in G.degree()]

    # Interactive Plot for Degree Distribution
    fig_degree = px.histogram(
        degrees,
        nbins=20,
        title="Degree Distribution",
        labels={'value': 'Degree', 'count': 'Frequency'},
        color_discrete_sequence=["#636EFA"]
    )
    fig_degree.update_layout(
        xaxis_title="Degree",
        yaxis_title="Frequency",
        bargap=0.1
    )
    fig_degree.show("png")

    # 4. Graph Visualization
    print("Graph Visualization")

    # Get node positions for visualization
    pos = nx.spring_layout(G, seed=42)

    # Extract node positions for Plotly
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    edge_x = []
    edge_y = []

    # Extract edges for Plotly
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    # Create edge traces
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            size=10,
            color=list(dict(G.degree()).values()),  # Use degree as node color
            colorscale='Viridis',
            colorbar=dict(
                title="Node Degree"
            ),
            line_width=2
        )
    )

    # Add hover info for nodes
    node_text = [f"Node: {node}<br>Degree: {deg}" for node, deg in G.degree()]
    node_trace.text = node_text

    # Create the figure
    fig_graph = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Interactive Graph of Predicted Links",
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False)
                        ))

    # Show the interactive graph
    fig_graph.show("png")
    
def Visualization_categories_distribution(df_links_,df_categories):
    df_links_source=df_links_.copy()
    df_links_source = df_links_source.merge(df_categories[['Article', 'Category_Level_1']], 
                            left_on='Source', right_on='Article', how='left')

    # Rename Source categories
    df_links_source = df_links_source.rename(columns={
        'Category_Level_1': 'Source_Category_1',
    })
    df_links_target=df_links_.copy()
    # Merge categories for the Target nodes
    df_links_target = df_links_target.merge(df_categories[['Article', 'Category_Level_1']], 
                            left_on='Target', right_on='Article', how='left')

    # Rename Target categories
    df_links_target = df_links_target.rename(columns={
        'Category_Level_1': 'Target_Category_1',
    })

    # Combine all categories (Source and Target) into one column
    source_columns = ['Source_Category_1']
    target_columns = ['Target_Category_1']

    source_categories = pd.concat([df_links_source[col] for col in source_columns], ignore_index=True)
    target_categories = pd.concat([df_links_target[col] for col in target_columns], ignore_index=True)
    all_categories = pd.concat([source_categories,target_categories])

    # Count the frequency of categories 
    category_counts = all_categories.value_counts().reset_index()
    category_counts.columns = ['Category', 'Frequency']
    source_category_counts = source_categories.value_counts().reset_index()[:10]
    source_category_counts.columns = ['Category', 'Frequency']
    target_category_counts = target_categories.value_counts().reset_index()[:10]
    target_category_counts.columns = ['Category', 'Frequency']

    # Drop NaN values
    #all_categories = all_categories.dropna()

    # Create an interactive bar chart using Plotly
    fig = px.bar(
        category_counts,
        x='Category',
        y='Frequency',
        title='Distribution of Categories for Predicted Links',
        labels={'Frequency': 'Count', 'Category': 'Category'},
        hover_data={'Category': True, 'Frequency': True},
        text='Frequency',
        color='Frequency',
        color_continuous_scale=px.colors.sequential.Sunset
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        xaxis=dict(tickangle=45, title='Category'),
        yaxis=dict(title='Frequency'),
        template='plotly_dark',
        title_font_size=20,
        title_x=0.5,
        margin=dict(l=50, r=50, t=80, b=150),
        height=600,
        width=1000
    )
    fig.show("png")

    # Plot Source Category Distribution using Plotly
    fig_source = px.bar(
        source_category_counts,
        x='Category',
        y='Frequency',
        title='Distribution of Categories for Source Nodes',
        labels={'Frequency': 'Count', 'Category': 'Category'},
        hover_data={'Category': True, 'Frequency': True},
        text='Frequency',
        color='Frequency',
        color_continuous_scale=px.colors.sequential.Sunset
    )

    fig_source.update_traces(texttemplate='%{text}', textposition='outside')
    fig_source.update_layout(
        xaxis=dict(tickangle=45, title='Category'),
        yaxis=dict(title='Frequency'),
        template='plotly_dark',
        title_font_size=20,
        title_x=0.5,
        margin=dict(l=50, r=50, t=80, b=150),
        height=600,
        width=1000
    )
    fig_source.show("png")

    # Plot Target Category Distribution using Plotly
    fig_target = px.bar(
        target_category_counts,
        x='Category',
        y='Frequency',
        title='Distribution of Categories for Target Nodes',
        labels={'Frequency': 'Count', 'Category': 'Category'},
        hover_data={'Category': True, 'Frequency': True},
        text='Frequency',
        color='Frequency',
        color_continuous_scale=px.colors.sequential.Sunset
    )

    fig_target.update_traces(texttemplate='%{text}', textposition='outside')
    fig_target.update_layout(
        xaxis=dict(tickangle=45, title='Category'),
        yaxis=dict(title='Frequency'),
        template='plotly_dark',
        title_font_size=20,
        title_x=0.5,
        margin=dict(l=50, r=50, t=80, b=150),
        height=600,
        width=1000
    )
    fig_target.show("png")

def visualization_pie_charts(df_links_, df_categories):
    df_links_pie=df_links_.copy()

    # Merge categories for the Source nodes
    df_links_pie = df_links_pie.merge(df_categories[['Article', 'Category_Level_1']], 
                            left_on='Source', right_on='Article', how='left')

    # Rename Source categories
    df_links_pie = df_links_pie.rename(columns={
        'Category_Level_1': 'Source_Category',
    })

    # Merge categories for the Target nodes
    df_links_pie = df_links_pie.merge(df_categories[['Article', 'Category_Level_1']], 
                            left_on='Target', right_on='Article', how='left')

    # Rename Target categories
    df_links_pie = df_links_pie.rename(columns={
        'Category_Level_1': 'Target_Category',
    })

    # Drop the duplicate 'Article' column after the second merge
    df_links_pie = df_links_pie.drop(columns=['Article_x','Article_y'])

    # Step 1: Identify the 4 most common source categories
    top_source_categories = (
        df_links_pie['Source_Category']
        .value_counts()
        .head(4)
        .index.tolist()
    )

    # Step 2: Filter the dataset for rows where the source category is in the top 10
    filtered_links = df_links_pie[df_links_pie['Source_Category'].isin(top_source_categories)]

    # Step 3: Group by Source_Category and Target_Category to calculate percentages
    redirect_counts = (
        filtered_links.groupby(['Source_Category', 'Target_Category'])
        .size()
        .reset_index(name='Count')
    )

    # Step 4: Normalize counts to percentages within each source category
    redirect_counts['Percentage'] = (
        redirect_counts.groupby('Source_Category')['Count']
        .transform(lambda x: x / x.sum() * 100)
    )

    # Step 5: Create one pie chart per source category
    for source_category in top_source_categories:
        # Filter data for the current source category
        data = redirect_counts[redirect_counts['Source_Category'] == source_category]

        # Create a pie chart using Plotly
        fig = px.pie(
            data,
            names='Target_Category',
            values='Percentage',
            title=f'Redirections from Articles belonging to {source_category} category',
            color='Target_Category',
            color_discrete_sequence=px.colors.sequential.Sunset
        )

        # Enhance layout and interactivity
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(
            title_font_size=18,
            title_x=0.5,
            height=600,
            width=800
        )

        # Show the pie chart
        fig.show("png")

def Visualization_error_bars(df_links_, df_categories, df_links_with_pred):
    # Merge categories for Source and Target nodes (consider all levels)
    df_links_with_pred = df_links_with_pred.merge(
        df_categories[['Article', 'Category_Level_1']],
        left_on='Source',
        right_on='Article',
        how='inner'
    ).rename(columns={
        'Category_Level_1': 'Source_Category_1',
    })

    df_links_with_pred = df_links_with_pred.merge(
        df_categories[['Article', 'Category_Level_1']],
        left_on='Target',
        right_on='Article',
        how='inner'
    ).rename(columns={
        'Category_Level_1': 'Target_Category_1'
    })

    # Drop the extra 'Article' columns from the merges
    df_links_with_pred = df_links_with_pred.drop(columns=['Article_x', 'Article_y'])


    # Combine all categories into a single column by creating DataFrames
    source_categories = df_links_with_pred[['Source_Category_1']].copy()
    source_categories['Correct_Label'] = df_links_with_pred['Prediction'] == df_links_with_pred['Correct_Label']
    source_categories = source_categories.rename(columns={'Source_Category_1': 'Category'})

    target_categories = df_links_with_pred[['Target_Category_1']].copy()
    target_categories['Correct_Label'] = df_links_with_pred['Prediction'] == df_links_with_pred['Correct_Label']
    target_categories = target_categories.rename(columns={'Target_Category_1': 'Category'})

    # Combine source and target categories
    all_categories = pd.concat([source_categories, target_categories], ignore_index=True)

    # Group by category and calculate the counts of correct and incorrect predictions
    category_stats = all_categories.groupby('Category')['Correct_Label'] \
                                    .value_counts(normalize=False) \
                                    .unstack(fill_value=0)

    # Normalize to get percentages
    category_stats_normalized = category_stats.div(category_stats.sum(axis=1), axis=0)
    category_stats_normalized.to_csv("Bar_data")
    # Limit the number of categories (e.g., top 20 based on total predictions)
    top_categories = category_stats.sum(axis=1).nlargest(10).index
    category_stats_top = category_stats.loc[top_categories]
    category_stats_normalized_top = category_stats_top.div(category_stats_top.sum(axis=1), axis=0)
    
    #Export to csv
    category_stats_normalized_top.to_csv("Data_error_bars.csv")
    # Plot as a horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    category_stats_normalized_top.sort_values(by=False, ascending=False).plot(
        kind='barh',
        stacked=True,
        ax=ax,
        color=['orange', 'skyblue'],
        edgecolor='black'
    )
    plt.title('Prediction Accuracy by Individual Categories', fontsize=16)
    plt.xlabel('Proportion', fontsize=14)
    plt.ylabel('Category', fontsize=14)
    plt.legend(title='Prediction Correct', labels=['Incorrect', 'Correct'], fontsize=12)
    plt.tight_layout()
    plt.show()
