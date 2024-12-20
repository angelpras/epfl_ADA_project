
# Linkin nodes

This is the project repository of the group collectifmetisser. To see the data story, go to https://n1naa.github.io

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

- Can article names and descriptions embeddings be used to infer a link between two articles? ***Answered in part 7***
- What are the most descriptive features/heuristic methods to infer links between two articles? ***Answered in part 7***
- Do additional links provided by a missing link predictor model aid in the human navigation between articles? ***Answered in part 9***
- Can the GNN model effectively handle large-scale Wikipedia article networks, or are there scalability challenges that need to be addressed for real-world use? ***Answered in part 7***
- Are there specific types of articles or categories where link prediction is more or less accurate? ***Answered in part 8***

## Methods

### Feature engineering

To maximize the performance of our GNN, we designed features that capture various aspects of the graph structure. Of course, there are other interesting ways to design features, that could be studied if the model's performance is found to be unsatisfactory.

#### Node (articles) features

- PageRank: Assigns a ranking score to each node, indicating its relative importance in the network.
- Eigenvector Centrality: Measures a node's influence within the graph based on its connections.
- Text Embeddings: We embed article titles and descriptions using vector representations.

#### Edge features

- Cosine similarity: Computed using the previously generated embeddings for articles and descriptions, it provides a measure of how similar two words or sentences are.
- Jaccard Similarity: Measures the proportion of shared neighbors between two nodes.
- Adamic-Adar Index: A weighted sum of shared neighbors, placing more weight on less-connected nodes.
- Preferential Attachment: Predicts links based on the degree of the nodes.

### Training/Validation/Testing Sample Choice

We will begin by using existing links as positive examples, labeled as 1. To identify unconnected pairs that are unlikely to represent missing links, we apply a **negative likelihood score**. This score helps select unconnected pairs that are more likely to be true non-links, labeling them as 0. By leveraging feature distributions, this approach effectively classifies unconnected pairs as negative examples.

1. **Distribution-Based Scoring**: Each feature (node distance, content similarity, common neighbors) is analyzed for its distribution among connected and unconnected pairs. Some examples are:
   - **Node Distance**: Connected pairs generally have a lower average distance.
   - **Content Similarity**: Connected pairs often exhibit higher content similarity scores.
   - **Common Neighbors**: Connected pairs usually share a greater number of common neighbors.

2. **Candidate Likelihood Calculation**: Based on the feature distributions, pairs which exhibit higher scores than the average for connected nodes are selected as candidates for link prediction.

3. **Threshold for Negative Examples**: The most dissimilar pairs, based on feature distributions, are assigned a 0 label as negative samples. A fixed number of these pairs are selected to create a balanced dataset with an equal number of 1 label connections already present.

### Graph Convolutional Network

The model uses a Graph Convolutional Network (GCN) to learn patterns within the graph. Node features are concatenated to create enriched representations, while edge features are concatenated separately to capture relationships. These features pass through multiple GCN layers, and a Multi-Layer Perceptron (MLP) generates a score between 0 and 1, indicating link likelihood between nodes.

Other models, like Graph Attention Networks (GATs) and GraphSAGE, were considered. GATs offer attention mechanisms for feature weighting but are computationally expensive, and edge features already provide importance weighting. GraphSAGE, designed for evolving graphs, wasn’t needed for this fixed graph.

## Contribution within the team

- Alexis: Building and coding the model architecture. Helping with 0 label non-links and candidates selection. Training the model. Doing the model ablation studies and plots.
- Antoine: Post-model analysis ideas, algorithms, and plots.
- Angeline: Distribution analysis pre-model for 0 label non-links and candidates selection. Post model graph statistics analysis.
- Nina: Coming up with the datastory: ideas, form, text, story, and making a lot of graphs.
- Alfred: Human path data analysis post-model. Making a lot of graphs for the datastory.

## Results Highlights

- Demonstrated the expressiveness of article and description embeddings for link prediction.
- Reduced mean lengths of human traces from 6.76 nodes to 6.65 nodes.
- Reduced mean game duration by 2.5 seconds, and the longest game by over 9 hours!