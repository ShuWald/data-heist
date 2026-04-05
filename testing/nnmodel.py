import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class GeoPredictor(nn.Module):

    def __init__(self, input_size):
        super(GeoPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 80),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(40, 2) 
        )
    
    def forward(self, x):
        return self.network(x)


class NNTrainer:

    def __init__(self, input_size, learning_rate=0.001, device='cpu'):
        self.device = device
        self.model = GeoPredictor(input_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.loss_fn(predictions, y_batch)
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.loss_fn(predictions, y_batch)
                val_loss += loss.item()
        
        avg_loss = val_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the model
        
        Args:
            X_train: Training features (numpy array or tensor)
            y_train: Training targets (numpy array or tensor)
            X_val: Validation features
            y_val: Validation targets
            epochs: Max number of training epochs
            batch_size: Batch size for training
        """
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        print(f"Training complete!")
    
    def predict(self, X):

        self.model.eval()
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.model(X)
        
        return predictions.cpu().numpy()