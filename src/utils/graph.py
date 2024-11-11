import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def create_graph(embedded_articles,df_links):
    """
    Returns G, the connected graph of the selected articles  
    """
    G = nx.DiGraph()

    # Add nodes with embeddings as attributes 
    for article, embedding in embedded_articles.items():
        G.add_node(article, embedding=embedding)


    # Add edeges to the graph
    for index, row in df_links.iterrows():
        article = row['Articles']
        links = row['Links']
        for link in links:
            embedding_article = embedded_articles.get(article)
            embedding_link = embedded_articles.get(link)
            
            if embedding_article is not None and embedding_link is not None:
                # Compute cosine similarity
                cosine_score = cosine_similarity(embedding_article.reshape(1, -1), embedding_link.reshape(1, -1))[0, 0]
                G.add_edge(article, link, weight=cosine_score)
    return G
    