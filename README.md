
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

## Methods

### Feature engineering

#### Node (articles) features

#### Edge features

#### Graph features

###

### 
First, we will use the existing links as positive examples, labeling them as 1. For unconnected article pairs, we will calculate a "negative likelihood" score based on features like content similarity, common neighbors, and node distance. Pairs with a high negative likelihood score will be treated as negative examples, labeled as 0. The remaining unconnected pairs will be left as unknown.

By training the GNN model on this mixture of positive and negative examples, it will learn to distinguish likely links from unlikely ones. The trained model can then be applied to the full set of article pairs, including the unknown connections, to predict the probability of a link. This approach allows us to uncover missing links that could enhance the coherence and navigability of the Wikipedia knowledge graph, while avoiding the biases inherent in manually curated links.