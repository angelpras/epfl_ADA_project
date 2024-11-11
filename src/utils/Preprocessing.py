import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
%matplotlib inline
import seaborn as sns
import math
import os
from urllib.parse import unquote
import re
from tqdm.autonotebook import tqdm, trange

from sentence_transformers import SentenceTransformer, util

from sklearn.metrics.pairwise import cosine_similarity

def decoding(df,column): 
    """
    Decode the URL encoding in the files names
    """
    df[column]=df[column].apply(lambda x: unquote(x) 
                            if (re.compile(r"%[0-9A-Fa-f]{2}").search(x)) else x )

def preprocessing_and_embedding():
    """
    Do the full preprocessing of the data and the embedding.
    Returns the cleaned dataframe for the articles and the links files
    as well as the articles embedded in two different ways (glove and sb)
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data')
    change_name_files_plaintext_articles(data_path)
    
    df_articles=preprocessing_articles(data_path)    
    df_links=preprocessing_links(data_path) 
    
    article_to_embedding=glove_embedding(df_articles)
    article_to_embedding2=sb_model_embedding(df_articles)
    
    return df_articles,df_links,article_to_embedding,article_to_embedding2



def change_name_files_plaintext_articles(data_path):
    """
    Modify the names of the files in the plaintext_articles folder, to 
    handle the url encoding
    """
    #Get the articles list
    plain_path=os.path.join(data_path,r"plaintext_articles")
    articles_list=os.listdir(plain_path)

    #Check if there is a # followed by two hexadecimal numbers (URL ENCODING FOR SPECIAL CHARACTERS)
    url_encoded_pattern = re.compile(r"%[0-9A-Fa-f]{2}")

    #Change the file name in the plaintext directory
    for article in articles_list:
        if url_encoded_pattern.search(article):
            if os.path.exists(unquote(os.path.join(plain_path,article))):
                print(f"Skipping rename, file already exists: {unquote(os.path.join(plain_path,article))}")
            else:
                print(f"Renamed: {os.path.join(plain_path,article)} -> {os.path.join(plain_path,article)}")
                os.rename(os.path.join(plain_path,article),unquote(os.path.join(plain_path,article)))
       
def preprocessing_articles(data_path):  
    """      
    Returns a dataframe corresponding to the article.tsv file,
    remove missing values, duplicates and handle the url encoding 
    """
    
    articles_path=os.path.join(data_path,r"wikispeedia_paths-and-graph","articles.tsv")
    df_articles = pd.read_csv(articles_path, sep='\t', header=None,comment="#")

    df_articles.columns=["Articles"]
    df_articles.index.name="Index"

    df_articles.dropna()
    if not df_articles["Articles"].is_unique:
        df_articles.drop_duplicates(subset=["Articles"], inplace=True)

    decoding(df_articles,"Articles")
    return df_articles
    
def  preprocessing_links(data_path):
    """      
    Returns a dataframe corresponding to the links.tsv file,
    remove missing values and handle the url encoding 
    """
    
    links_path=os.path.join(data_path,r"wikispeedia_paths-and-graph","links.tsv")
    df_links = pd.read_csv(links_path, comment="#", sep="\t", header=None)
    df_links.columns=["Articles","Links"]

    df_links.dropna()

    decoding(df_links,"Articles")
    decoding(df_links,"Links")
    return df_links

def glove_embedding(df_articles,device_="cpu"):
    
    glove_model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d', device=device_)
    list_articles = df_articles["Articles"].tolist()
    embeddings_articles = glove_model.encode(list_articles, convert_to_numpy =True)
    df_articles["Embeddings"] = list(embeddings_articles)
    article_to_embedding = dict(zip(list_articles, embeddings_articles))
    return article_to_embedding
    
    
def sb_model_embedding(df_articles,device_="cpu"):
    
    sb_model = SentenceTransformer("all-MiniLM-L6-v2", device=device_)
    list_articles = [articles.replace("_", " ") for articles in list_articles]
    embeddings_articles2 = sb_model.encode(list_articles, convert_to_numpy =True)
    df_articles["Embeddings"] = list(embeddings_articles2)
    article_to_embedding2 = dict(zip(list_articles, embeddings_articles2))
    return article_to_embedding2
