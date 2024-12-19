import numpy as np
import pandas as pd
import networkx as nx
import torch
import random
from typing import Dict, Tuple, List
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sentence_transformers.util import dot_score
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity

def articles_to_embeddings(parsed_articles, model, embeddings_path):
    """
    Returns a dictionary with the article titles as keys and the embeddings of the title and description as values
    """
    df = pd.DataFrame(parsed_articles, columns=['Article_Title', 'Related_Subjects', 'Description'])
    df['Article_Title_embedding'] = df['Article_Title'].apply(model.encode, engine='numba', engine_kwargs={'parallel':True})
    df['Description_embedding'] = df['Description'].apply(model.encode, engine='numba', engine_kwargs={'parallel':True})
    df.to_pickle(embeddings_path)
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

def add_edges_from_csv(G, linked_nodes):
    """
    Adds edges to the graph G based on a CSV containing Source and Target articles.

    Args:
        G: The original graph (nx.DiGraph).
        linked_nodes: additional edges to add to the graph.        
    Returns:
        G: The updated graph with additional edges.
    """
    # Iterate through each row in the additional edges
    for _, row in linked_nodes.iterrows():
        source = row['Source']
        target = row['Target']
        G.add_edge(source, target)
    
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

def calculate_labels_cos_similarity(G, similarities):
    """
    Identifies candidate node pairs for link prediction based on cosine similarity thresholds.
    
    Args:
        G: The graph object containing connected nodes with title and description similarity weights.
        similarities: Dictionary containing unconnected pairs with their cosine similarities.
    
    Returns:
        candidates: Set of unconnected node pairs whose average similarity equals or exceeds
                   the mean similarity of existing connected pairs. These pairs are considered
                   candidates for link prediction.
    """
    # Extract unconnected pairs with cosine similarities
    unconnected_similarities = similarities['unconnected_pairs']
    
    # Initialize set for candidate pairs
    candidates = set()
    
    # Calculate average similarity across all connected pairs to use as threshold
    connected_scores = []
    for u, v, data_edge in G.edges(data=True):
        title_similarity = data_edge['weight_title']
        description_similarity = data_edge['weight_description']
        average_similarity = 0.5 * title_similarity + 0.5 * description_similarity
        connected_scores.append(average_similarity)
    min_threshold_candidates = sum(connected_scores) / len(connected_scores)
    
    # Identify unconnected pairs that exceed the threshold as candidates
    for pair in unconnected_similarities:
        source, target = pair['source'], pair['target']
        title_similarity = pair['title_similarity']
        description_similarity = pair['description_similarity']
        average_similarity = 0.5 * title_similarity + 0.5 * description_similarity
        if average_similarity >= min_threshold_candidates:
            candidates.add((source, target))
            
    return candidates

def calculate_labels_jaccard(jaccard_scores, zero_label_non_links):
    """
    Filters a set of non-linked node pairs based on their Jaccard similarity scores.
    
    Args:
        jaccard_scores: Dictionary containing Jaccard coefficients for unconnected pairs
                       in the format {'unconnected_scores': [{'source': node1, 
                       'target': node2, 'score': float}]}.
        zero_label_non_links: Initial set of node pairs labeled as non-links to be filtered.
    
    Returns:
        filtered_zero_label_non_links: Subset of the input non-links where each pair's
                                     Jaccard score is less than or equal to the maximum
                                     Jaccard score observed across all unconnected pairs.
    """
    # Get list of Jaccard scores for unconnected pairs
    jaccard_unconnected_scores = [entry['score'] for entry in jaccard_scores['unconnected_scores']]
    maximal_unconnected_score = max(jaccard_unconnected_scores)
    
    # Keep only non-links whose Jaccard score doesn't exceed the maximum observed score
    filtered_zero_label_non_links = set(
        (source, target) for source, target in zero_label_non_links
        if next(
            (entry['score'] for entry in jaccard_scores['unconnected_scores']
             if entry['source'] == source and entry['target'] == target),
            0
        ) <= maximal_unconnected_score
    )
    
    return filtered_zero_label_non_links

def create_zero_label_non_links(similarities, target_size=120000):
    """
    Creates a set of node pairs that are least likely to form links, based on their
    combined title and description cosine similarities.
    
    Args:
        similarities: Dictionary containing unconnected pairs with their title and
                     description similarities in format {'unconnected_pairs': 
                     [{'source': node1, 'target': node2, 'title_similarity': float,
                     'description_similarity': float}]}.
        target_size: Maximum number of pairs to include in the result set. 
                    Defaults to 120000.
    
    Returns:
        zero_label_non_links: Set of (source, target) tuples representing the pairs
                             with the lowest average cosine similarities, limited to
                             the specified target size.
    """
    # Get list of unconnected node pairs
    unconnected_similarities = similarities['unconnected_pairs']
    
    # Sort pairs by average similarity (title and description) in ascending order
    sorted_unconnected = sorted(
        unconnected_similarities,
        key=lambda x: 0.5 * x['title_similarity'] + 0.5 * x['description_similarity']
    )
    
    # Take the first target_size pairs (those with lowest similarities)
    zero_label_non_links = set(
        (pair['source'], pair['target']) for pair in sorted_unconnected[:target_size]
    )
    
    return zero_label_non_links


def node2index_maps(embedded_articles):
    """
    Returns a dictionary mapping node names to their corresponding indices
    """
    nodes = list(embedded_articles.keys())
    nodes.sort()
    node_to_index = {node: idx for idx, node in enumerate(nodes)}
    index_to_node = {idx: node for node, idx in node_to_index.items()}
    
    return node_to_index, index_to_node

class GraphDataLoader:
    def __init__(self, graph: nx.DiGraph, candidates: set=None,
                 zero_label_non_links: set=None, feature_to_drop: List[str]=[]):
        """
        Initialize the data loader with a directed graph
        
        Args:
            graph (nx.DiGraph): Input graph representing article connections
        """
        self.graph = graph
        self.candidates = candidates
        self.zero_label_non_links = zero_label_non_links
        self.feature_to_drop = feature_to_drop
        
        self.node_features = {}
        self.edge_features = {}

        self.nodes = list(graph.nodes())
        self.nodes.sort()
        self.node_to_index = {node: idx for idx, node in enumerate(self.nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}

    def compute_node_features(self) -> None:
        """
        Compute various node-level features
        """
        # PageRank
        pagerank = nx.pagerank(self.graph)
        
        # Eigenvector Centrality
        eigenvector_centrality = nx.eigenvector_centrality(self.graph)
        
        # Store features
        for node in self.nodes:
            # Retrieve title and description embeddings
            title_embedding = self.graph.nodes[node].get('embedding_title', None)
            desc_embedding = self.graph.nodes[node].get('embedding_description', None)
            
            node_features = {
                'title_embedding': title_embedding,
                'description_embedding': desc_embedding,
                'pagerank': pagerank.get(node, 0),
                'eigenvector_centrality': eigenvector_centrality.get(node, 0)
            }

            # Filter out features listed in feature_to_drop
            self.node_features[node] = {
                feature: value for feature, value in node_features.items() if feature not in self.feature_to_drop
            }

    def compute_edge_features(self) -> None:
        """
        Compute edge-level features using NetworkX built-in functions
        
        Features include:
        - Jaccard Similarity
        - Adamic-Adar Index
        - Preferential Attachment
        - Existing graph-created similarities
        """
        for u, v, data in self.graph.edges(data=True):
            # Existing similarities from graph creation
            title_similarity = data.get('weight_title', 0)
            description_similarity = data.get('weight_description', 0)

            # Compute Jaccard Similarity (for directed graph)
            u_neighbors = set(self.graph.successors(u))
            v_neighbors = set(self.graph.successors(v))
            
            # Compute Adamic-Adar Index
            common_neighbors = u_neighbors.intersection(v_neighbors)
            adamic_adar_index = sum(
                1 / np.log(max(1, self.graph.out_degree(neighbor) + self.graph.in_degree(neighbor))) 
                for neighbor in common_neighbors
            )
            
            # Compute Preferential Attachment Score
            u_degree = self.graph.out_degree(u) + self.graph.in_degree(u)
            v_degree = self.graph.out_degree(v) + self.graph.in_degree(v)
            preferential_attachment = u_degree * v_degree

            jaccard_sim = 0
            if u_neighbors or v_neighbors:
                jaccard_sim = len(u_neighbors.intersection(v_neighbors)) / len(u_neighbors.union(v_neighbors))
            
            edge_features = {
                'title_similarity': title_similarity,
                'description_similarity': description_similarity,
                'jaccard_similarity': jaccard_sim,
                'adamic_adar_index': adamic_adar_index,
                'preferential_attachment': preferential_attachment,
            }

            # Filter out features listed in feature_to_drop
            self.edge_features[(u, v)] = {
                feature: value for feature, value in edge_features.items() if feature not in self.feature_to_drop
            }

    def create_pyg_dataset(self) -> Data:
        """
        Create a PyTorch Geometric dataset with node and edge features
        
        Returns:
            Data: PyTorch Geometric data object
        """
        # Compute node and edge features if not already done
        if not self.node_features:
            self.compute_node_features()
        
        if not self.edge_features:
            self.compute_edge_features()
        
        # Get positive links
        positive_links = list(self.graph.edges())
        positive_links.sort() # Sort for reproducibility

        # Get negative samples
        negative_links = list(self.zero_label_non_links)
        negative_links.sort() # Sort for reproducibility
        
        # Convert links to index-based representation
        indexed_positive_links = [(self.node_to_index[u], self.node_to_index[v]) for u, v in positive_links]
        indexed_negative_links = [(self.node_to_index[u], self.node_to_index[v]) for u, v in negative_links]
        
        # Combine links with labels
        all_links = indexed_positive_links + indexed_negative_links
        labels = [1] * len(positive_links) + [0] * len(negative_links)
        
        # Create edge index and labels tensors
        edge_index = torch.tensor(all_links, dtype=torch.long).t().contiguous()
        edge_labels = torch.tensor(labels, dtype=torch.float)
        
        # Get candidate links
        candidate_links = list(self.candidates)
        
        # Convert links to index-based representation
        indexed_candidate_links = [(self.node_to_index[u], self.node_to_index[v]) for u, v in candidate_links]
        
        # Create edge index and labels tensors
        candidates_edge_index = torch.tensor(indexed_candidate_links, dtype=torch.long).t().contiguous()

        # Create node feature tensor with robust handling of embeddings
        node_features = []
        for node in self.nodes:
            # Extract features, converting to list and handling potential None values
            title_embedding = self.node_features[node].get('title_embedding', [0] * 384)  # Default 384-dim zero vector
            desc_embedding = self.node_features[node].get('description_embedding', [0] * 384)
            pagerank = [self.node_features[node].get('pagerank', 0)]
            eigenvector = [self.node_features[node].get('eigenvector_centrality', 0)]
            
            # Flatten and convert to numpy/torch compatible format
            node_feature = (
                (title_embedding if isinstance(title_embedding, list) else title_embedding.tolist()) +
                (desc_embedding if isinstance(desc_embedding, list) else desc_embedding.tolist()) +
                pagerank +
                eigenvector
            )
            
            node_features.append(node_feature)
        
        # Convert node features to tensor
        node_features = torch.tensor(node_features, dtype=torch.float)

        # Create edge feature tensor
        edge_features = torch.tensor([
            [
                self.edge_features.get((u, v), {}).get('title_similarity', 0),
                self.edge_features.get((u, v), {}).get('description_similarity', 0),
                self.edge_features.get((u, v), {}).get('jaccard_similarity', 0),
                self.edge_features.get((u, v), {}).get('adamic_adar_index', 0),
                self.edge_features.get((u, v), {}).get('preferential_attachment', 0)
            ] for u, v in all_links
        ], dtype=torch.float)

        candidates_edge_features = torch.tensor([
            [
                self.edge_features.get((u, v), {}).get('title_similarity', 0),
                self.edge_features.get((u, v), {}).get('description_similarity', 0),
                self.edge_features.get((u, v), {}).get('jaccard_similarity', 0),
                self.edge_features.get((u, v), {}).get('adamic_adar_index', 0),
                self.edge_features.get((u, v), {}).get('preferential_attachment', 0)
            ] for u, v in candidate_links
        ], dtype=torch.float)

        # Create PyG Data object
        data = Data(
            x=node_features, 
            edge_index=edge_index, 
            edge_attr=edge_features, 
            y=edge_labels
        )
        data_candidates = Data(
            x=node_features,
            edge_index=candidates_edge_index,
            edge_attr=candidates_edge_features
        )

        return data, data_candidates
        
def create_edge_datasets(dataset, candidates_dataset, train_ratio=0.7, val_ratio=0.15):
    """
    Split the dataset into training, validation, and test sets
    
    Args:
        data (Data): Original PyG Data object
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation
    
    Returns:
        list: List of individual edge Data objects
    """
    # Total number of edges
    total_edges = dataset.edge_index.shape[1]
    
    # Shuffle edge indices
    shuffle_idx = torch.randperm(total_edges)
    shuffled_edge_index = dataset.edge_index[:, shuffle_idx]
    shuffled_edge_attr = dataset.edge_attr[shuffle_idx]
    shuffled_labels = dataset.y[shuffle_idx]
    
    # Calculate split indices
    train_end = int(total_edges * train_ratio)
    val_end = train_end + int(total_edges * val_ratio)
    
    # Create individual edge datasets
    edge_datasets = []
    for i in range(total_edges):
        edge_data = Data(
            x=dataset.x,
            edge_index=shuffled_edge_index[:, i:i+1],
            edge_attr=shuffled_edge_attr[i:i+1],
            y=shuffled_labels[i:i+1]
        )
        edge_datasets.append(edge_data)

    candidates_edge_datasets = []
    for i in range(len(candidates_dataset.edge_index[1])):
        edge_data = Data(
            x=candidates_dataset.x,
            edge_index=candidates_dataset.edge_index[:, i:i+1],
            edge_attr=candidates_dataset.edge_attr[i:i+1]
        )
        candidates_edge_datasets.append(edge_data)

    return {
        'train': edge_datasets[:train_end],
        'val': edge_datasets[train_end:val_end],
        'test': edge_datasets[val_end:],
        'candidates': candidates_edge_datasets
    }

def create_graph_dataloaders(dataset, candidates_dataset, batch_size=32):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data (Data): PyTorch Geometric Data object
        batch_size (int): Batch size for dataloaders
    
    Returns:
        tuple: Train, validation, and test dataloaders
    """
    edge_datasets = create_edge_datasets(dataset, candidates_dataset)
    train_loader = DataLoader(edge_datasets['train'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(edge_datasets['val'], batch_size=batch_size)
    test_loader = DataLoader(edge_datasets['test'], batch_size=batch_size)
    candidates_loader = DataLoader(edge_datasets['candidates'], batch_size=batch_size)

    return train_loader, val_loader, test_loader, candidates_loader


