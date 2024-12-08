import torch
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

from models.GCN_model import *
from data.Graph import *
from data.Preprocessing import *


def train_gcn(model, train_loader, val_loader, criterion, optimizer, 
              device, epochs=50, early_stopping_patience=10):
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

def evaluate_model(model, test_loader, device):
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
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = (outputs > 0.5).float()
            
            all_preds.append(preds)
            all_labels.append(batch.y)
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).float().mean()
    precision = (all_preds[all_preds == 1] == all_labels[all_preds == 1]).float().mean()
    recall = (all_preds[all_labels == 1] == all_labels[all_labels == 1]).float().mean()
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }

# Example usage
def main():
    data_path = os.path.abspath(os.path.join(os.getcwd(), 'data'))
    df_links = preprocessing_links(data_path)

    embeddings_path = os.path.join(data_path, 'embeddings.pkl')
    if os.path.exists(embeddings_path):
        df = pd.read_pickle(embeddings_path)
        embedded_articles = dict(zip(df['Article_Title'], zip(df['Article_Title_embedding'], df['Description_embedding'] )))
    else:
        print("Couldn't find the embeddings")
    
    print("Creating Graph")
    G=create_graph(embedded_articles, df_links)
    print("Making data loaders")
    data_loader = GraphDataLoader(G)
    train_loader, val_loader, test_loader = data_loader.create_graph_dataloaders()

    # Assume data and loaders are prepared as before
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    print("Initializing model")
    model = EdgeClassificationGCNWrapper().to(device)
    
    # Loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train the model
    print("Starting training")
    trained_model = train_gcn(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device
    )
    
    # Evaluate on test set
    metrics = evaluate_model(trained_model, test_loader, device)
    print("Test Metrics:", metrics)

if __name__ == '__main__':
    main()
