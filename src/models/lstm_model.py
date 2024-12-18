# project/src/models/lstm_model.py (Modified for classification and confidence scoring)

import torch
import torch.nn as nn
import numpy as np
import os
from src.models.base_model import BaseModel

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        out = self.fc(output[:, -1, :])
        out = self.sigmoid(out)
        return out

class LSTMModel(BaseModel):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1, learning_rate=1e-3, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTMClassifier(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.criterion = nn.BCELoss()  # Binary cross entropy for classification
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        train_size = X_train_t.size(0)
        for epoch in range(epochs):
            # Shuffle training data
            permutation = torch.randperm(train_size)
            X_train_t = X_train_t[permutation]
            y_train_t = y_train_t[permutation]

            self.model.train()
            epoch_loss = 0.0
            for i in range(0, train_size, batch_size):
                X_batch = X_train_t[i:i+batch_size]
                y_batch = y_train_t[i:i+batch_size]

                self.optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze(-1)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= (train_size / batch_size)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(X_val_t).squeeze(-1)
                    val_loss = self.criterion(val_preds, y_val_t).item()
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy().squeeze()
        # preds are probabilities. We can use them as is for confidence scoring.
        return preds  # confidence score is the probability

    def save(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
