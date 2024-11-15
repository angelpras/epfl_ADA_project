import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers.util import dot_score
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity

def articles_to_embeddings(parsed_articles, model):
    """
    Returns a dictionary with the article titles as keys and the embeddings of the title and description as values
    """
    df = pd.DataFrame(parsed_articles, columns=['Article_Title', 'Related_Subjects', 'Description'])
    df['Article_Title_embedding'] = df['Article_Title'].apply(model.encode, engine='numba', engine_kwargs={'parallel':True})
    df['Description_embedding'] = df['Description'].apply(model.encode, engine='numba', engine_kwargs={'parallel':True})
    embedded_articles = dict(zip(df['Article_Title'], zip(df['Article_Title_embedding'], df['Description_embedding'] )))
    
    return embedded_articles

def create_graph(embedded_articles, df_links):
    """
    Returns G, the connected graph of the selected articles with the features:
    - Nodes: Article titles, with attributes for the embeddings of the title and description
    - Edges: Links between articles, with weights based on cosine similarity of the embeddings
    """
    G = nx.DiGraph()

    # Add nodes with embeddings as attributes 
    for article, (embedding_title, embedding_description) in embedded_articles.items():
        G.add_node(article, embedding_title=embedding_title,embedding_description=embedding_description)


    # Add edges to the graph
    for _, row in df_links.iterrows():
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
    In this function, these graph characteristics are computed and displayed:
    - Number of nodes and edges
    - Average degree
    - Degree distribution
    - Network density
    - Clustering coefficient
    - Average shortest path length
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


def Node2Vec_func(G):
    """
    Generates node embeddings by simulating biased random walks through a graph
    """
    #The parameters below would be tuned, especially p and q to balance the DFS and BFS behavior
    node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, p=1, q=1, workers=4)

    # Learn the embeddings for the nodes in the graph
    model = node2vec.fit()

    # Get the node embeddings (for each node)
    embeddings = model.wv

    # Extract node embeddings and store them in a DataFrame
    node_embeddings = []

    for node in G.nodes():
        try:
            # Retrieve the embedding for the node
            embedding = embeddings[node]
            node_embeddings.append({"Article": node, "Embedding": embedding})
        except KeyError:
            # Handle case where node embedding might be missing
            node_embeddings.append({"Article": node, "Embedding": None})

    # Convert to DataFrame
    df_embeddings = pd.DataFrame(node_embeddings)
    
    # Handle missing embeddings if any
    df_embeddings = df_embeddings.dropna(subset=["Embedding"])

    # Flatten the embeddings into individual columns in one step
    embedding_matrix = pd.DataFrame(
        df_embeddings["Embedding"].to_list(),
        columns=[f"embedding_{i}" for i in range(128)],
        index=df_embeddings.index
    )
    
    # Concatenate with the original DataFrame and drop the "Embedding" column
    df_embeddings = pd.concat([df_embeddings.drop(columns=["Embedding"]), embedding_matrix], axis=1)
    
    # Compute the cosine similarity between all pairs of nodes
    embedding_matrix_values = df_embeddings[[f"embedding_{i}" for i in range(128)]].values
    cosine_sim_matrix = cosine_similarity(embedding_matrix_values)
    
    return cosine_sim_matrix, df_embeddings

