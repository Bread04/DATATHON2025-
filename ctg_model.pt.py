# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 23:02:32 2025

@author: braed
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load and clean your CTG data
df = pd.read_excel("CTG.xlsx", sheet_name=1, header=1)
redundant = ['b', 'e', 'AC.1', 'FM.1', 'UC.1', 'DL.1', 'DS.1', 'DP.1',
             'CLASS', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']
df = df.dropna(axis=1).drop_duplicates().drop(columns=redundant, errors='ignore')

# Feature selection
numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != 'NSP']
X = df[numeric_cols]
y = df['NSP']

X_scaled = MinMaxScaler().fit_transform(X)
selected = SelectKBest(score_func=chi2, k=10).fit(X_scaled, y)
top_features = [numeric_cols[i] for i in selected.get_support(indices=True)]

# Final training data
X_final = df[top_features].values
y_final = df['NSP'].values
X_final = StandardScaler().fit_transform(X_final)

X_train, _, y_train, _ = train_test_split(X_final, y_final, test_size=0.2, stratify=y_final)

# Define model
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

# Train model
model = CTGClassifier(input_dim=X_final.shape[1])
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

# Save model
torch.save(model.state_dict(), "ctg_model.pt")
print("✅ Model saved as ctg_model.pt")

