import torch
import sys
import os
import pickle
import argparse
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from models.GCN_model import *
from data.Graph import *
from data.Preprocessing import *
from utils.Visualization import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GCN model for link prediction with ablation studies')
    
    # Define possible node and edge features
    node_features = [
        'title_embedding',
        'description_embedding',
        'pagerank',
        'eigenvector_centrality'
    ]
    edge_features = [
        "title_similarity",
        "description_similarity",
        "jaccard_similarity",
        "adamic_adar_index",
        "preferential_attachment"
    ]
    
    # Add argument to specify features to remove
    parser.add_argument(
        '--features_to_drop',
        nargs='+',  # Allows specifying multiple features as a list
        choices=node_features + edge_features,  # Restrict valid inputs to defined features
        default=[],  # Default to no features removed
        help=(
            "Specify features to remove for ablation study. "
            "Node features: {node}. "
            "Edge features: {edge}."
            .format(
                node=", ".join(node_features),
                edge=", ".join(edge_features)
            )
        )
    )
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save the model')
    return parser.parse_args()

args = parse_args()

def train_gcn(model, train_loader, val_loader, criterion, optimizer, 
              device, epochs=20, early_stopping_patience=10):
    """
    Training function for the GCN model
    
    Args:
        model (torch.nn.Module): GCN model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimization algorithm
        device (torch.device): Computing device
        epochs (int): Number of training epochs
        early_stopping_patience (int): Epochs to wait for improvement
    
    Returns:
        torch.nn.Module: Trained model
    """
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            batch = batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                val_loss = criterion(outputs, batch.y)
                total_val_loss += val_loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_gcn_model.pth'))
        else:
            patience_counter += 1
        
        # Stop training if no improvement
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_gcn_model.pth'), weights_only=True))
    return model

def evaluate_model(model, data_loader, device, index_to_node, threshold=0.5, candidate=False):
    """
    Evaluate the model on data and generate predictions. Calculates metrics if labels are available.
    
    Args:
        model (torch.nn.Module): Trained GCN model
        data_loader (DataLoader): Data loader
        device (torch.device): Computing device
        index_to_node (dict): Mapping from node indices to node names
        threshold (float): Classification threshold
    
    Returns:
        dict: Evaluation metrics (if labels available) or None
    """
    model.eval()
    all_preds = []
    all_labels = []
    link_nodes = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, leave=False):
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs > threshold).float()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            # Store predictions and labels if available
            all_preds.append(preds)
            if not candidate:
                all_labels.append(batch.y)
            
            # Process each prediction in the batch
            for idx in range(preds.shape[0]):
                try:
                    source_idx = batch[idx].edge_index[0].item()
                    target_idx = batch[idx].edge_index[1].item()
                    
                    source_node = index_to_node.get(source_idx, f"Unknown Node {source_idx}")
                    target_node = index_to_node.get(target_idx, f"Unknown Node {target_idx}")
                    
                    if not candidate:
                        link_nodes.append((source_node, target_node, preds[idx].item(), batch.y[idx].item()))
                    else:
                        link_nodes.append((source_node, target_node, preds[idx].item()))
                except Exception as e:
                    print(f"Error processing link at index {idx}: {e}")
    
    # Save results to CSV
    if link_nodes:
        columns = ['Source', 'Target', 'Prediction']
        if not candidate:
            columns.append('Correct_Label')
            df_links = pd.DataFrame(link_nodes, columns=columns)
            df_links.to_csv(os.path.join(args.save_dir, 'linked_nodes_with_predictions.csv'), index=False)
        if candidate:
            df_links = pd.DataFrame(link_nodes, columns=columns)
            df_links = df_links[df_links['Prediction'] == 1]
            df_links = df_links.drop(columns='Prediction')
            df_links.to_csv(os.path.join(args.save_dir, 'linked_nodes.csv'), index=False)
        print("CSV saved successfully.")
    else:
        print("No links were processed. CSV is empty.")
    
    # Calculate metrics only if labels are available
    if all_labels:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        print("First 40 predictions")
        print(all_preds[:40])
        print("First 40 labels")
        print(all_labels[:40])
        
        accuracy = (all_preds == all_labels).float().mean()
        precision = (all_preds[all_preds == 1] == all_labels[all_preds == 1]).float().mean()
        recall = (all_preds[all_labels == 1] == all_labels[all_labels == 1]).float().mean()
        f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }
    
    return None

# Example usage
def main():
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    df_links = preprocessing_links(data_path)

    embeddings_path = os.path.join(data_path, 'embeddings.pkl')
    if os.path.exists(embeddings_path):
        df = pd.read_pickle(embeddings_path)
        embedded_articles = dict(zip(df['Article_Title'], zip(df['Article_Title_embedding'], df['Description_embedding'] )))
    else:
        print("Couldn't find the embeddings")

    save_dir = args.save_dir
    graph_data_path = os.path.join(save_dir, 'graph_dataset.pkl')
    candidates_data_path = os.path.join(save_dir, 'candidates_dataset.pkl')

    if os.path.exists(graph_data_path) and os.path.exists(candidates_data_path):
        with open(graph_data_path, 'rb') as f:
            dataset = pickle.load(f)

        with open(candidates_data_path, 'rb') as f:
            candidates_dataset = pickle.load(f)

    else:
        print("Creating Graph")
        G=create_graph(embedded_articles, df_links)
        unconnected_pairs = create_subset_unconnected_nodes(G)
        similarities = create_node_similarity_distributions(G, unconnected_pairs)
        jaccard_scores = calculate_jaccards_coeff(G, unconnected_pairs, plot=False)

        # Create the zero-label non-links from article title and description similarities
        zero_label_non_links = create_zero_label_non_links(similarities)
        print(f"Number of zero-label non-links: {len(zero_label_non_links)}")
        # Use jacard scores to filter out non-links that are too similar.
        zero_label_non_links = calculate_labels_jaccard(jaccard_scores, zero_label_non_links)
        print(f"Number of zero-label non-links after filtering: {len(zero_label_non_links)}")
        
        candidates = calculate_labels_cos_similarity(G, similarities)

        data_loader = GraphDataLoader(G, candidates, zero_label_non_links, args.features_to_drop)
        dataset, candidates_dataset = data_loader.create_pyg_dataset()
        with open(graph_data_path, 'wb') as f:
            pickle.dump(dataset, f)

        with open(candidates_data_path, 'wb') as f:
            pickle.dump(candidates_dataset, f)
    

    dataset = dataset.to(device, non_blocking=True)
    candidates_dataset = candidates_dataset.to(device, non_blocking=True)

    print("Making data loaders")
    train_loader, val_loader, test_loader, candidates_loader = create_graph_dataloaders(dataset, candidates_dataset)
    
    # Initialize model
    print("Initializing model")
    model = EdgeClassificationGCNWrapper().to(device)

    if os.path.exists(os.path.join(args.save_dir, 'best_gcn_model.pth')):
        # If model already trained, load it
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_gcn_model.pth'), map_location=device, weights_only=True))
    else:
        # Loss and optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Train the model
        print("Starting training")
        model = train_gcn(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            device
        ) 

    _, index_to_node = node2index_maps(embedded_articles)
    # Evaluate on test set
    print("Evaluating on test set")
    metrics = evaluate_model(model, test_loader, device, index_to_node, threshold=0.9)
    print("Test Metrics:", metrics)

    print("Evaluating on the candidates set")
    evaluate_model(model, candidates_loader, device, index_to_node, threshold=0.9, candidate=True)


if __name__ == '__main__':
    main()
