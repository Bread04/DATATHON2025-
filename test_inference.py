# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 22:51:31 2025

@author: braed
"""
# Test Inference 
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_exploration.data_cleanup import load_and_clean_data, select_top_features
from train_model import CTGClassifier

def infer(filepath, model_path='ctg_model.pt'):
    df = load_and_clean_data(filepath)
    features = select_top_features(df)
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = CTGClassifier(input_dim=X.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor)
        predicted_classes = torch.argmax(predictions, dim=1).numpy()

    df['Predicted_NSP'] = predicted_classes
    print(df[['NSP', 'Predicted_NSP']].head())

if __name__ == "__main__":
    infer("path/to/your/ctg_data.xlsx")
