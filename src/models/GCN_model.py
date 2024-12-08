import torch
import torch.nn.functional as F
from torch_geometric.nn.models import GCN
import torch.nn as nn

class EdgeClassificationGCNWrapper(torch.nn.Module):
    """
    Wrapper for PyG's built-in GCN to handle edge classification
    """
    def __init__(self, 
                 node_feature_dim=770,   # Number of input node features
                 hidden_channels=64,     # Hidden layer dimensions
                 num_layers=3,           # Number of GCN layers
                 dropout_rate=0.3,       # Dropout rate
                 edge_feature_dim=5):    # Number of edge features
        """
        Initialize the GCN model with edge classification head
        
        Args:
            node_feature_dim (int): Dimension of input node features
            hidden_channels (int): Number of hidden channels in GCN layers
            num_layers (int): Number of GCN layers
            dropout_rate (float): Dropout probability
            edge_feature_dim (int): Dimension of edge features
        """
        super().__init__()
        
        # Initial node feature embedding
        self.node_embedding = nn.Linear(node_feature_dim, hidden_channels)
        
        # Built-in PyG GCN model
        self.gcn = GCN(
            in_channels=hidden_channels, 
            hidden_channels=hidden_channels, 
            num_layers=num_layers,
            out_channels=hidden_channels,
            dropout=dropout_rate
        )
        
        # Edge feature processing
        self.edge_feature_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Edge classification head
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # Binary classification
        )
    
    def forward(self, data):
        """
        Forward pass of the edge classification model
        
        Args:
            data (Data): PyTorch Geometric data object
        
        Returns:
            torch.Tensor: Predicted edge probabilities
        """
        # Initial node feature transformation
        x = self.node_embedding(data.x)
        
        # Apply GCN
        x = self.gcn(x, data.edge_index)
        
        # Process edge features
        edge_features = self.edge_feature_mlp(data.edge_attr)
        
        # Get source and target node embeddings for each edge
        src_nodes = x[data.edge_index[0]]
        dst_nodes = x[data.edge_index[1]]
        
        # Concatenate node embeddings and edge features
        edge_repr = torch.cat([src_nodes, dst_nodes, edge_features], dim=1)
        
        # Classify edges
        edge_prob = self.classifier(edge_repr).squeeze()
        
        return edge_prob