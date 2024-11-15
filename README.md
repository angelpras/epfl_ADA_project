
# Linkin nodes

This is the project repository of the group collectifmetisser.

## Quickstart

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-collectifmetisser
cd ada-2024-project-collectifmetisser

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```

## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

## Abstract

The goal of this project is to develop a graph neural network (GNN) model that can effectively predict missing links between Wikipedia articles. The motivation behind this work is to address the inherent biases and inconsistencies present in the existing link structure of Wikipedia. Often, the links between articles are created manually by human editors, leading to incomplete or inconsistent connections between related concepts.

By training the GNN model to infer likely links based on article content and structural properties, we aim to uncover hidden connections that should exist but are currently missing. This systematic approach to link prediction can help improve the overall organization and discoverability of information within the Wikipedia knowledge base.

Notably, it will aid in the navigation between articles, as users can more easily discover relevant content related to their current area of interest.

## Research questions

- Can article names and descriptions embeddings be used to infer a link between two articles?
- What are the most descriptive features/heuristic methods to infer links between two articles?
- Do additional links provided by a missing link predictor model aid in the human navigation between articles?
- Can the GNN model effectively handle large-scale Wikipedia article networks, or are there scalability challenges that need to be addressed for real-world use?
- Are there specific types of articles or categories where link prediction is more or less accurate?

## Methods

### Feature engineering

To maximize the performance of our GNN, we designed features that capture various aspects of the graph structure. Of course, there are other interesting ways to design features, that could be studied if the model's performance is found to be unsatisfactory.

#### Node (articles) features

- PageRank: Assigns a ranking score to each node, indicating its relative importance in the network.
- Eigenvector Centrality: Measures a node's influence within the graph based on its connections.
- Text Embeddings: We embed article titles and descriptions using vector representations.
- Common Neighbors: Quantifies the overlap in the neighborhood between pairs of nodes.

#### Edge features

- Cosine similarity: Computed using the previously generated embeddings for articles and descriptions, it provides a measure of how similar two words or sentences are.
- Jaccard Similarity: Measures the proportion of shared neighbors between two nodes.
- Adamic-Adar Index: A weighted sum of shared neighbors, placing more weight on less-connected nodes.
- Preferential Attachment: Predicts links based on the degree of the nodes.

#### Graph features

- Node2Vec: Generates node embeddings that capture graph structure and connectivity by exploring random walks.

### Training/Validation/Testing Sample Choice

We will begin by using existing links as positive examples, labeled as 1. To identify unconnected pairs that are unlikely to represent missing links, we apply a **negative likelihood score**. This score helps select unconnected pairs that are more likely to be true non-links, labeling them as 0. By leveraging feature distributions, this approach effectively classifies unconnected pairs as negative examples.

1. **Distribution-Based Scoring**: Each feature (node distance, content similarity, common neighbors) is analyzed for its distribution among connected and unconnected pairs. Some examples are:
   - **Node Distance**: Connected pairs generally have a lower average distance.
   - **Content Similarity**: Connected pairs often exhibit higher content similarity scores.
   - **Common Neighbors**: Connected pairs usually share a greater number of common neighbors.

2. **Negative Likelihood Score Calculation**: Based on the feature distributions, a **negative likelihood score** is assigned to unconnected pairs. This score weights each feature according to its contribution to low connection likelihood.

3. **Threshold for Negative Examples**: A threshold is set for the negative likelihood score. Unconnected pairs with scores above this threshold are designated as negative examples (label 0) during training and validation, helping the model learn distinctions between likely and unlikely connections. A second threshold is then also set where all the unconnected pairs with scores below this threshold are considered as candidates for link prediction.

### Graph Convolutional Network

The model uses a Graph Convolutional Network (GCN) to learn patterns within the graph. Node features are concatenated to create enriched representations, while edge features are concatenated separately to capture relationships. These features pass through multiple GCN layers, and a Multi-Layer Perceptron (MLP) generates a score between 0 and 1, indicating link likelihood between nodes.

Other models, like Graph Attention Networks (GATs) and GraphSAGE, were considered. GATs offer attention mechanisms for feature weighting but are computationally expensive, and edge features already provide importance weighting. GraphSAGE, designed for evolving graphs, wasn’t needed for this fixed graph.

## Proposed Timeline

- **15-22 Nov**: Focus on completing Homework 2.
- **23-29 Nov**: 
  - Finish Homework 2.
  - Start building the dataloaders and defining the model architecture.
  - Build a test graph by connecting random nodes together.
- **30 Nov - 6 Dec**: 
  - Train the model using various configurations and feature sets.
  - Setup the pipeline to compare graphs together using the human traces.
- **7-13 Dec**: 
  - Perform tests on the new graphs created with human navigation traces.
- **14-20 Dec**: 
  - Apply final adjustments and complete final touches.

## Organization within the team

- Alexis and Antoine: Build the model architecture, using the concatenated embeddings. Train the model.
- Angeline and Nina: Build the dataloaders by building the negative likelihood score calculator and set the thresholds for label 0 (non-link) and candidates for missing link prediction. Determine the most useful features for the model, potentially making coefficients for the negative likelihood score calculator.
- Alfred: Will work on graph evaluation. Builds the test graph to compare the orignal graph and have a pipeline in place. Use human traces to evaluate graph performance.

## Questions for TAs

- Is the model choice good?
- What others features could be implemented ? 
- Is our method for the Training/Validation samples and candidates choice for prediction a good approach or are there some other methods?
