import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers.util import dot_score

def articles_to_embeddings(parsed_articles, model):
    
    df = pd.DataFrame(parsed_articles, columns=['Article_Title', 'Related_Subjects', 'Description'])
    df['Article_Title_embedding'] = df['Article_Title'].apply(model.encode, engine='numba', engine_kwargs={'parallel':True})
    df['Description_embedding'] = df['Description'].apply(model.encode, engine='numba', engine_kwargs={'parallel':True})
    embedded_articles = dict(zip(df['Article_Title'], zip(df['Article_Title_embedding'], df['Description_embedding'] )))
    
    return embedded_articles

def create_graph(embedded_articles, df_links):
    """
    Returns G, the connected graph of the selected articles  
    """
    G = nx.DiGraph()

    # Add nodes with embeddings as attributes 
    for article, (embedding_title, embedding_description) in embedded_articles.items():
        G.add_node(article, embedding_title=embedding_title,embedding_description=embedding_description)


    # Add edges to the graph
    for index, row in df_links.iterrows():
        article = row['Articles']
        links = row['Links']
        for link in links:
            embedding_article = embedded_articles.get(article)
            embedding_link = embedded_articles.get(link)
            
            if embedding_article is not None and embedding_link is not None:
                # Compute cosine similarity
                embedding_title_article = embedding_article[0]
                embedding_description_article = embedding_article[1]
                embedding_title_link = embedding_link[0]
                embedding_description_link = embedding_link[1]

                cosine_title = float(dot_score(embedding_title_article, embedding_title_link))
                cosine_description = float(dot_score(embedding_description_article, embedding_description_link))

                G.add_edge(article, link, weight_title=cosine_title, weight_description=cosine_description)
            else:
                print(f"Article {article} or {link} couldn't be found")

    return G 

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
    plt.ylabel("Occurances")
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


def weisfeiler_lehman_step(graph, labels):
    """Perform one WL iteration on the graph and return updated labels."""
    new_labels = {}
    for node in graph.nodes():
        # Create a multi-set label combining the node's current label and its neighbors' labels
        neighborhood = [labels[neighbor] for neighbor in graph.neighbors(node)]
        neighborhood.sort()
        new_labels[node] = hash((labels[node], tuple(neighborhood)))
    return new_labels