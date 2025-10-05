# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:50:11 2025

@author: braed
"""
# Train The Model
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_exploration.data_cleanup import load_and_clean_data, select_top_features

class CTGClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.net(x)

def train(filepath, save_path='ctg_model.pt'):
    df = load_and_clean_data(filepath)
    features = select_top_features(df)
    X = df[features].values
    y = df['NSP'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y)

    model = CTGClassifier(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
