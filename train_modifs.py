import torch
import sys
import os
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from models.GCN_model import *
from data.Graph import *
from data.Preprocessing import *
from utils.Visualization import *

def train_gcn(model, train_loader, val_loader, criterion, optimizer, 
              device, epochs=2, early_stopping_patience=10):
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
            torch.save(model.state_dict(), 'best_gcn_model.pth')
        else:
            patience_counter += 1
        
        # Stop training if no improvement
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_gcn_model.pth'))
    return model

def evaluate_model(model, test_loader, device, index_to_node, threshold = 0.5):
    """
    Evaluate the trained model on test data
    
    Args:
        model (torch.nn.Module): Trained GCN model
        test_loader (DataLoader): Test data loader
        device (torch.device): Computing device
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs > threshold).float()
            
            all_preds.append(preds)
            all_labels.append(batch.y)
            link_nodes=[]
            # Collect node names for links
            for idx in range(preds.shape[0]):
                try:
                    # Safely get the node indices
                    source_idx = batch[idx].edge_index[0].item()
                    target_idx = batch[idx].edge_index[1].item()
                  
                    # Look up nodes, using a default if not found
                    source_node = index_to_node.get(source_idx, f"Unknown Node {source_idx}")
                    target_node = index_to_node.get(target_idx, f"Unknown Node {target_idx}")

                    link_nodes.append((source_node, target_node, preds[idx],batch.y[idx]))
                except Exception as e:
                    print(f"Error processing link at index {idx}: {e}")
            

    # Save results to CSV
    if link_nodes:
        df_links = pd.DataFrame(link_nodes, columns=['Source', 'Target','Prediction', 'Correct_Prediction'])
        df_links.to_csv('linked_nodes_with_predictions.csv', index=False)
        print("CSV saved successfully.")
    else:
        print("No links were processed. CSV is empty.")
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    print("First 40 predictions")
    print(all_preds[:40])
    print("First 40 labels")
    print(all_labels[:40])
    # Calculate metrics
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

def model_infer(model, test_loader, index_to_node, device, threshold=0.95):
    """
    Evaluate the trained model on test data and output predictions with correctness.

    Args:
        model (torch.nn.Module): Trained GCN model
        test_loader (DataLoader): Test data loader
        index_to_node (dict): Mapping from node indices to node names
        device (torch.device): Computing device
        threshold (float): Threshold for classification

    Returns:
        tuple: Predictions and link nodes with correctness
    """
    model.eval()
    all_preds = []
    link_nodes = []

    with torch.no_grad():
        for batch in tqdm(test_loader, leave=False):
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs > threshold).float()

            # Find where predictions are links (1.0)
            link_indices = torch.where(preds == 1)[0]
            
            # Collect node names for links
            for idx in link_indices:
                try:
                    # Safely get the node indices
                    source_idx = batch[idx].edge_index[0].item()
                    target_idx = batch[idx].edge_index[1].item()
                                
                    # Look up nodes, using a default if not found
                    source_node = index_to_node.get(source_idx, f"Unknown Node {source_idx}")
                    target_node = index_to_node.get(target_idx, f"Unknown Node {target_idx}")

                    link_nodes.append((source_node, target_node,1))
                except Exception as e:
                    print(f"Error processing link at index {idx}: {e}")
            
            link_indices = torch.where(preds == 0)[0]
            
            # Collect node names for links
            for idx in link_indices:
                try:
                    # Safely get the node indices
                    source_idx = batch[idx].edge_index[0].item()
                    target_idx = batch[idx].edge_index[1].item()
                 
                    # Look up nodes, using a default if not found
                    source_node = index_to_node.get(source_idx, f"Unknown Node {source_idx}")
                    target_node = index_to_node.get(target_idx, f"Unknown Node {target_idx}")

                    link_nodes.append((source_node, target_node,0))
                except Exception as e:
                    print(f"Error processing link at index {idx}: {e}")
            all_preds.append(preds)

    # Save results to CSV
    if link_nodes:
        df_links = pd.DataFrame(link_nodes, columns=['Source', 'Target', 'Correct_Prediction'])
        df_links.to_csv('linked_nodes_with_predictions.csv', index=False)
        print("CSV saved successfully.")
    else:
        print("No links were processed. CSV is empty.")

    return all_preds, link_nodes


# Example usage
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    df_links = preprocessing_links(data_path)

    embeddings_path = os.path.join(data_path, 'embeddings.pkl')
    if os.path.exists(embeddings_path):
        df = pd.read_pickle(embeddings_path)
        embedded_articles = dict(zip(df['Article_Title'], zip(df['Article_Title_embedding'], df['Description_embedding'] )))
    else:
        print("Couldn't find the embeddings")

    if os.path.exists('graph_dataset.pkl') and os.path.exists('candidates_dataset.pkl'):
        with open('graph_dataset.pkl', 'rb') as f:
            dataset = pickle.load(f)

        with open('candidates_dataset.pkl', 'rb') as f:
            candidates_dataset = pickle.load(f)

    else:
        print("Creating Graph")
        G=create_graph(embedded_articles, df_links)
        similarities = create_node_similarity_distributions(G)
        candidates, zero_label_non_links = calculate_negative_likelihood_and_labels(G, similarities)

        data_loader = GraphDataLoader(G, candidates, zero_label_non_links)
        dataset, candidates_dataset = data_loader.create_pyg_dataset()
        with open('graph_dataset.pkl', 'wb') as f:
            pickle.dump(dataset, f)

        with open('candidates_dataset.pkl', 'wb') as f:
            pickle.dump(candidates_dataset, f)
    

    dataset = dataset.to(device, non_blocking=True)

    print("Making data loaders")
    train_loader, val_loader, test_loader, candidates_loader = create_graph_dataloaders(dataset, candidates_dataset)
    
    # Initialize model
    print("Initializing model")
    model = EdgeClassificationGCNWrapper().to(device)

    if os.path.exists('best_gcn_model.pth'):
        # If model already trained, load it
        model.load_state_dict(torch.load('best_gcn_model.pth', map_location=torch.device('cpu'), weights_only=True))

        # Move the model to the appropriate device (CUDA or CPU)
        model = model.to(device)

    # else:
    # Loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    _, index_to_node = node2index_maps(embedded_articles)

    # # Evaluate on test set
    print("evaluating on test set")
    metrics = evaluate_model(model, test_loader, device, index_to_node, threshold=0.5)
    print("Test Metrics:", metrics)

    
    print("Evaluating on the candidates set")
    preds, linked_nodes = model_infer(model, candidates_loader, index_to_node, device, threshold=0.9)

if __name__ == '__main__':
    main()
