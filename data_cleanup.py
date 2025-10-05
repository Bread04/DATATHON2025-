# -*- coding: utf-8 -*-
# data_exploration/data_cleanup.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

def load_and_clean_data(filepath):
    df = pd.read_excel(filepath, sheet_name=1, header=1)

    # Drop NaNs and duplicates
    df = df.dropna(axis=1).drop_duplicates()

    # Drop symbolic and redundant features
    redundant = ['b', 'e', 'AC.1', 'FM.1', 'UC.1', 'DL.1', 'DS.1', 'DP.1',
                 'CLASS', 'A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']
    df = df.drop(columns=redundant, errors='ignore')

    return df

def visualize_class_distribution(df, target='NSP'):
    class_counts = df[target].value_counts().sort_index()
    class_props = class_counts / class_counts.sum()

    sns.barplot(x=class_counts.index.astype(int), y=class_counts.values)
    for i, val in enumerate(class_counts.values):
        label = f"{val}\n{class_props.iloc[i]:.1%}"
        plt.text(i, val + 5, label, ha='center')
    plt.title("NSP Class Distribution")
    plt.xlabel("NSP Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def select_top_features(df, target='NSP', k=10):
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != target]
    X = df[numeric_cols]
    y = df[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_scaled, y)

    selected = [numeric_cols[i] for i in selector.get_support(indices=True)]
    return selected

def reduce_dimensions(df, features, target='NSP'):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df[target])
    plt.title("PCA Projection of CTG Data")
    plt.show()


