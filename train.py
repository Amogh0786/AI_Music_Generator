import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import MusicLSTM
import numpy as np

def train_model():
    print("Loading data...")
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
        
    network_input = data['network_input']
    network_output = data['network_output']
    n_vocab = data['n_vocab']
    
    print(f"Vocabulary size: {n_vocab}")
    
    # Convert to PyTorch tensors
    X = torch.tensor(network_input, dtype=torch.long)
    y = torch.tensor(network_output, dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MusicLSTM(n_vocab=n_vocab)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), 'music_model.pth')
    print("Model saved to music_model.pth")

if __name__ == '__main__':
    train_model()
