import numpy as np
import pandas as pd
import networkx as nx
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

    return G 
    