import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools
import seaborn as sns
import pandas as pd
import numpy as np

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

def visualize_connected_node_similarity_distributions(G):

    weight_titles = [data['weight_title'] for _, _, data in G.edges(data=True)]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(weight_titles, bins=50, edgecolor='black') 
    plt.title('Distribution of Cosine Similarity in Article Titles Between Connected Nodes')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Number of Occurrences')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    weight_description = [data['weight_description'] for _, _, data in G.edges(data=True)]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(weight_description, bins=50, edgecolor='black') 
    plt.title('Distribution of Cosine Similarity in Article Description Between Connected Nodes')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Number of Occurrences')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def visualize_unconnected_node_similarity_distributions(G, subset_size=350):
    all_nodes = list(G.nodes)
    subset_nodes = random.sample(all_nodes, subset_size)

    # Create a subgraph with the subset of nodes
    subgraph = G.subgraph(subset_nodes)
    connected_pairs = set()
    for u, v in subgraph.edges():
        connected_pairs.add((u, v))
        connected_pairs.add((v, u))

    # Find all unconnected article pairs in the subgraph
    all_pairs = set((a, b) for a in subset_nodes for b in subset_nodes if a != b)
    unconnected_pairs = all_pairs - connected_pairs

    # Calculate cosine similarities for unconnected pairs
    similarities = []
    for source, target in unconnected_pairs:
        embedding_title_source = subgraph.nodes[source]['embedding_title']
        embedding_description_source = subgraph.nodes[source]['embedding_description']
        embedding_title_target = subgraph.nodes[target]['embedding_title']
        embedding_description_target = subgraph.nodes[target]['embedding_description']
        
        cosine_title = float(dot_score(embedding_title_source, embedding_title_target))
        cosine_description = float(dot_score(embedding_description_source, embedding_description_target))

        similarities.append({
            'source': source,
            'target': target,
            'title_similarity': cosine_title,
            'description_similarity': cosine_description
        })

    # Visualization for connected nodes
    weight_titles = [s['title_similarity'] for s in similarities]
    weight_descriptions = [s['description_similarity'] for s in similarities]

    # Plot the histograms
    plt.figure(figsize=(10, 6))
    plt.hist(weight_titles, bins=50, edgecolor='black')
    plt.title('Distribution of Cosine Similarity in Article Titles Between Unconnected Nodes')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Number of Occurrences')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(weight_descriptions, bins=50, edgecolor='black')
    plt.title('Distribution of Cosine Similarity in Article Descriptions Between Unconnected Nodes')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Number of Occurrences')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    return similarities


def storing_cosine_similarties(G, subset_size=350):
    # Lists for storing cosine similarities
    connected_titles_similarities = []
    connected_descriptions_similarities = []
    unconnected_titles_similarities = []
    unconnected_descriptions_similarities = []

    # Cosine similarity for connected nodes
    for _, _, data in G.edges(data=True):
        connected_titles_similarities.append(data['weight_title'])
        connected_descriptions_similarities.append(data['weight_description'])

    # Cosine similarity for unconnected nodes
    all_nodes = list(G.nodes)
    subset_nodes = random.sample(all_nodes, subset_size)
    subgraph = G.subgraph(subset_nodes)

    connected_pairs = set()
    for u, v in subgraph.edges():
        connected_pairs.add((u, v))
        connected_pairs.add((v, u))

    # Find all unconnected article pairs in the subgraph
    all_pairs = set((a, b) for a in subset_nodes for b in subset_nodes if a != b)
    unconnected_pairs = all_pairs - connected_pairs

    # Calculate cosine similarities for unconnected pairs
    for source, target in unconnected_pairs:
        embedding_title_source = subgraph.nodes[source]['embedding_title']
        embedding_description_source = subgraph.nodes[source]['embedding_description']
        embedding_title_target = subgraph.nodes[target]['embedding_title']
        embedding_description_target = subgraph.nodes[target]['embedding_description']
        

        # for articles title
        cosine_title = cosine_similarity([embedding_title_source], [embedding_title_target])[0, 0]
        unconnected_titles_similarities.append(cosine_title)

        # for articles description
        cosine_description = cosine_similarity([embedding_description_source], [embedding_description_target])[0, 0]
        unconnected_descriptions_similarities.append(cosine_description)


    return {
        'connected_titles': connected_titles_similarities,
        'connected_descriptions': connected_descriptions_similarities,
        'unconnected_titles': unconnected_titles_similarities,
        'unconnected_descriptions': unconnected_descriptions_similarities
    }


def visualize_connected_vs_unconnected_cs_distribution(G):
    similarities = storing_cosine_similarties(G)
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


def calculate_links_conditional_proba(G):
    similarities = storing_cosine_similarties(G)
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
    plt.title('Estimated probability of a link between two random nodes according to cosine similary with articles descriptions')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(df_titles['bin_center'], df_titles['p(link|similarity)'], width=0.05, color='skyblue', edgecolor='black')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Estimated probability of a link between two random nodes')
    plt.title('Estimated probability of a link between two random nodes according to cosine similary with articles titles')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def calculate_preferential_attachment(G):
    ############ Connected Nodes ############
    G_connected_undirected = G.to_undirected()
    attachment_connected_vals = nx.preferential_attachment(G_connected_undirected)
    attachment_connected_scores = [p for _, _, p in attachment_connected_vals]

    ############ Unconnected Nodes ############
    subset_size = 350
    all_nodes = list(G.nodes)
    subset_nodes = random.sample(all_nodes, subset_size)

    # Create a subgraph with the subset of nodes
    subgraph = G.subgraph(subset_nodes)
    connected_pairs = set()
    for u, v in subgraph.edges():
        if u < v:
            connected_pairs.add((u, v))
        else:
            connected_pairs.add((v, u))

    # Find all unconnected article pairs in the subgraph
    all_pairs = set((a, b) for a in subset_nodes for b in subset_nodes if a < b)
    unconnected_pairs = all_pairs - connected_pairs
    # Remove direction and find attachment scores for unconnected nodes
    subgraph_undirected = subgraph.to_undirected()
    attachment_unconnected_scores = [score for _, _, score in nx.preferential_attachment(subgraph_undirected, ebunch=unconnected_pairs)]

    ############ Plotting Preferential Score Frequency and Values per Node Pairs ############
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Preferential Attachment Scores for Connected Pairs', fontsize=16)
    ax1.hist(attachment_connected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Preferential Attachment Scores for Node Pairs | Connected Pairs')
    ax1.set_xlabel('Preferential Attachment Score')
    ax1.set_ylabel('Frequency')
    ax2.scatter(range(len(attachment_connected_scores)), attachment_connected_scores, color='blue', alpha=0.5)
    ax2.set_title('Preferential Attachment Scores for Node Pairs | Connected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Preferential Attachment Score')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Preferential Attachment Scores for Unconnected Pairs', fontsize=16)
    ax1.hist(attachment_unconnected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Preferential Attachment Scores | Graph Subset | Unconnected Pairs')
    ax1.set_xlabel('Preferential Attachment Score')
    ax1.set_ylabel('Frequency')
    ax2.scatter(range(len(attachment_unconnected_scores)), attachment_unconnected_scores, color='blue', alpha=0.5)
    ax2.set_title('Preferential Attachment Scores for Node Pairs | Graph Subset | Unconnected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Preferential Attachment Score')
    plt.tight_layout()
    plt.show()

def calculate_common_neighbors(G):
    ############ Connected Nodes ############
    G_undirected = G.to_undirected()
    edges = G_undirected.edges()
    common_neighbors_connected_counts = []
    for u, v in edges:
        common_neighbors_connected_u_v = list(nx.common_neighbors(G_undirected, u, v))  
        common_neighbors_connected_counts.append(len(common_neighbors_connected_u_v))  

    ############ Unconnected Nodes ############
    subset_size = 350
    all_nodes = list(G.nodes)
    subset_nodes = random.sample(all_nodes, subset_size)

    # Create a subgraph with the subset of nodes
    subgraph = G.subgraph(subset_nodes) # It doesn't matter that this part is directional because the nb of common neighbors is a symetrical quantity
    connected_pairs = set()
    for u, v in subgraph.edges():
        if u < v:
            connected_pairs.add((u, v))
        else:
            connected_pairs.add((v, u))

    # Find all unconnected article pairs in the subgraph
    all_pairs = set((a, b) for a in subset_nodes for b in subset_nodes if a < b)
    unconnected_pairs = all_pairs - connected_pairs
    # Remove direction and find attachment scores for unconnected nodes
    subgraph_undirected = subgraph.to_undirected()
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
    ax2.hist(common_neighbors_unconnected_counts, bins=50, edgecolor='black')
    ax2.set_title('Distribution of Common Neighbors | Unconnected Pairs')
    ax2.set_xlabel('Number of Common Neighbors')
    ax2.set_ylabel('Frequency')
    ax2.set_xlim(0,np.max(common_neighbors_connected_counts))
    plt.tight_layout()
    plt.show()

def calculate_jaccards_coeff(G):
    ############ Connected Nodes ############
    G_connected_undirected = G.to_undirected()
    jaccard_connected_vals = nx.jaccard_coefficient(G_connected_undirected)
    jaccard_connected_scores = [j for _, _, j in jaccard_connected_vals]

    ############ Unconnected Nodes ############
    subset_size = 350
    all_nodes = list(G.nodes)
    subset_nodes = random.sample(all_nodes, subset_size)

    # Create a subgraph with the subset of nodes
    subgraph = G.subgraph(subset_nodes)
    connected_pairs = set()
    for u, v in subgraph.edges():
        if u < v:
            connected_pairs.add((u, v))
        else:
            connected_pairs.add((v, u))

    # Find all unconnected article pairs in the subgraph
    all_pairs = set((a, b) for a in subset_nodes for b in subset_nodes if a < b)
    unconnected_pairs = all_pairs - connected_pairs
    # Remove direction and find attachment scores for unconnected nodes
    subgraph_undirected = subgraph.to_undirected()
    jaccard_unconnected_scores = [score for _, _, score in nx.preferential_attachment(subgraph_undirected, ebunch=unconnected_pairs)]

    ############ Plotting Jaccard's Coefficient Frequency and Values per Node Pairs ############
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Jaccard\'s Coefficient for Connected Pairs', fontsize=16)
    ax1.hist(jaccard_connected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Jaccard\'s Coefficient for Node Pairs | Connected Pairs')
    ax1.set_xlabel('Jaccard\'s Coefficient')
    ax1.set_ylabel('Frequency')
    ax2.scatter(range(len(jaccard_connected_scores)), jaccard_connected_scores, color='blue', alpha=0.5)
    ax2.set_title('Jaccard\'s Coefficient for Node Pairs | Connected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Jaccard\'s Coefficient')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Jaccard\'s Coefficient for Unconnected Pairs', fontsize=16)
    ax1.hist(jaccard_unconnected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Jaccard\'s Coefficient | Graph Subset | Unconnected Pairs')
    ax1.set_xlabel('Jaccard\'s Coefficient')
    ax1.set_ylabel('Frequency')
    ax2.scatter(range(len(jaccard_unconnected_scores)), jaccard_unconnected_scores, color='blue', alpha=0.5)
    ax2.set_title('Jaccard\'s Coefficient for Node Pairs | Graph Subset | Unconnected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Jaccard\'s Coefficient')
    plt.tight_layout()
    plt.show()

def calculate_adamic_adar(G):
    ############ Connected Nodes ############
    G_connected_undirected = G.to_undirected()
    adar_connected_vals = nx.adamic_adar_index(G_connected_undirected)
    adar_connected_scores = [j for _, _, j in adar_connected_vals]

    ############ Unconnected Nodes ############
    subset_size = 350
    all_nodes = list(G.nodes)
    subset_nodes = random.sample(all_nodes, subset_size)

    # Create a subgraph with the subset of nodes
    subgraph = G.subgraph(subset_nodes)
    connected_pairs = set()
    for u, v in subgraph.edges():
        if u < v:
            connected_pairs.add((u, v))
        else:
            connected_pairs.add((v, u))

    # Find all unconnected article pairs in the subgraph
    all_pairs = set((a, b) for a in subset_nodes for b in subset_nodes if a < b)
    unconnected_pairs = all_pairs - connected_pairs
    # Remove direction and find attachment scores for unconnected nodes
    subgraph_undirected = subgraph.to_undirected()
    adar_unconnected_scores = [score for _, _, score in nx.adamic_adar_index(subgraph_undirected, ebunch=unconnected_pairs)]

    ############ Plotting Adamic/Adar Coefficient Frequency and Values per Node Pairs ############
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Adamic/Adar Coefficient for Connected Pairs', fontsize=16)
    ax1.hist(adar_connected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Adamic/Adar Coefficient for Node Pairs | Connected Pairs')
    ax1.set_xlabel('Adamic/Adar Coefficient')
    ax1.set_ylabel('Frequency')
    ax2.scatter(range(len(adar_connected_scores)), adar_connected_scores, color='blue', alpha=0.5)
    ax2.set_title('Adamic/Adar Coefficient for Node Pairs | Connected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Adamic/Adar Coefficient')
    plt.tight_layout()
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6)) 
    fig.suptitle('Adamic/Adar Coefficient for Unconnected Pairs', fontsize=16)
    ax1.hist(adar_unconnected_scores, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Distribution of Adamic/Adar Coefficient | Graph Subset | Unconnected Pairs')
    ax1.set_xlabel('Adamic/Adar Coefficient')
    ax1.set_ylabel('Frequency')
    ax2.scatter(range(len(adar_unconnected_scores)), adar_unconnected_scores, color='blue', alpha=0.5)
    ax2.set_title('Adamic/Adar Coefficient for Node Pairs | Graph Subset | Unconnected Pairs')
    ax2.set_xlabel('Node Pair Index')
    ax2.set_ylabel('Adamic/Adar Coefficient')
    plt.tight_layout()
    plt.show()

