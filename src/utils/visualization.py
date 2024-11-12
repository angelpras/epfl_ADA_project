import matplotlib.pyplot as plt
import networkx as nx
import random
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
        if u < v:
            connected_pairs.add((u, v))
        else:
            connected_pairs.add((v, u))

    # Find all unconnected article pairs in the subgraph
    all_pairs = set((a, b) for a in subset_nodes for b in subset_nodes if a < b)
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

