# DATATHON2025-
The main objective of this challenge is to use real-world CTG recordings to build a solution that supports clinicians in spotting fetal distress. By analyzing patterns in the data, you are expected to design a system that can reliably separate Normal, Suspect, and Pathologic cases. 

    from pathlib import Path
    import warnings

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    
    from scipy import stats
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.utils.class_weight import compute_class_weight
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        SMOTE = None
        warnings.warn("imblearn not available. Install with `pip install imbalanced-learn` to enable SMOTE demos.")
    
    sns.set_theme(style="whitegrid", palette="deep")
    pd.options.display.float_format = '{:,.2f}'.format
    
#Load Excel Data
    
    ctg_path = Path("C:/Users/Admin/Downloads/CTG.xlsx")  # Update if needed
    assert ctg_path.exists(), f"Expected Excel file at {ctg_path}"
    
#Open the Sheets inside the Excel File 
#Load Sheet 2 with header from row 2
    def read_feature_sheet(path: Path, sheet=1):
        xls = pd.ExcelFile(path)
        sheet_name = xls.sheet_names[sheet] if isinstance(sheet, int) else sheet
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=1)
        except ValueError:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
        return df
    
# Clean headers and data
# Drop empty columns and rows
# Reset index

    def clean_ctg_sheet(df):
        df = df.dropna(axis=1, how='all').dropna()
        df.reset_index(drop=True, inplace=True)
        return df
    
#Load and clean

    sheet2_raw = read_feature_sheet(ctg_path, sheet=1)
    sheet2_cleaned = clean_ctg_sheet(sheet2_raw)
    
#Display summary and first 1000 rows
    
    print("Shape:", sheet2_cleaned.shape)
    print("Columns:", sheet2_cleaned.columns.tolist())
    sheet2_cleaned.head(1000)
#Description of our information
    
    sheet2_shape = sheet2_cleaned.shape
    sheet2_info = sheet2_cleaned.info()
    sheet2_cleaned.describe(include='all').T.head(300)
    
    
    
#Setting the Distribution for NSP
    
# Count and proportion of each class
    class_counts = sheet2_cleaned['NSP'].value_counts().sort_index()
    class_props = class_counts / class_counts.sum()
    
# Bar plot of class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=class_counts.index.astype(int), y=class_counts.values, ax=ax)
    ax.set_xlabel('NSP Class')
    ax.set_ylabel('Count')
    ax.set_title('NSP Class Distribution')
    
# Annotate bars with count and proportion
    for index, value in enumerate(class_counts.values):
        label = f"{value}\n{class_props.iloc[index]:.1%}"
        ax.text(index, value + 5, label, ha='center')
    
    plt.tight_layout()
    plt.show()
    
# Tabular summary
    pd.DataFrame({
        'Class': class_counts.index.astype(int),
        'Count': class_counts.values,
        'Proportion': class_props.values
    })
    
# Display groupings
    for group, cols in feature_groups.items():
        print(f"\n🧠 {group} ({len(cols)} features):")
        for col in cols:
            print(f"• {col}")# Required import for Path
                  
    
# Define column groups
    physiological_signals = ['AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR']
    variability_metrics = ['ASTV', 'mSTV', 'ALTV', 'mLTV']
    histogram_descriptors = ['Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
    timing_markers = ['b', 'e']
    baseline_references = ['LB', 'LBE']
    redundant_features = ['b', 'e',' AC.1', 'FM.1', 'UC.1', 'DL.1', 'DS.1', 'DP.1', 'CLASS',' A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']
    symbolic_features = ['A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'CLASS',]
    target_variable = 'NSP'  # Use string, not list
    
# Organize into dictionary
    feature_groups = {
        'Physiological Signals': physiological_signals,
        'Variability Metrics': variability_metrics,
        'Histogram Descriptors': histogram_descriptors,
        'Timing Markers': timing_markers,
        'Baseline References': baseline_references,
        'Redundant Features': redundant_features,
        'Symbolic Features': symbolic_features,
        'Target Variable': [target_variable]
    }
    
    
# DATA CLEANING TO REMOVE NaNs in columns
    sheet2_filled = sheet2_cleaned.dropna(axis=1)
    print("Remaining NaNs:", sheet2_filled.isna().sum().sum())
    sheet2_filled.head()
    
# Remove duplicate rows
    duplicate_rows = sheet2_cleaned.duplicated().sum()
    print("Duplicate rows:", duplicate_rows)
    sheet2_cleaned = sheet2_cleaned.drop_duplicates()
    
# Drop label leakage and redundant columns
    sheet2_cleaned = sheet2_cleaned.drop(columns=redundant_features, errors='ignore')
    
# Define numeric columns (excluding dropped ones)
    numeric_cols = [col for col in sheet2_cleaned.columns if pd.api.types.is_numeric_dtype(sheet2_cleaned[col])]
    
# Prepare training data
    X = sheet2_cleaned[numeric_cols]
    y = sheet2_cleaned[target_variable]  # Use original sheet to retain target
    
    print("X shape:", X.shape)
    print("y distribution:\n", y.value_counts().sort_index())
    
    sheet2_cleaned.head()


## DATA CLEANING STEP 2

#NSP CLASS DISTRIBUTION

    class_counts = sheet2_cleaned['NSP'].value_counts().sort_index()
    class_props = class_counts / class_counts.sum()
    
    sns.barplot(x=class_counts.index.astype(int), y=class_counts.values)
    plt.title("NSP Class Distribution")
    plt.xlabel("NSP Class")
    plt.ylabel("Count")
    plt.show()
<img width="774" height="524" alt="image" src="https://github.com/user-attachments/assets/cf40f2a6-7746-4a78-9dd1-ea2fd863891f" />

#Feature Distribution By Class For ATV

    sns.violinplot(x='NSP', y='ASTV', data=sheet2_cleaned)
    plt.title("ASTV Distribution by NSP Class")
    plt.show()
    numeric_cols = [col for col in sheet2_cleaned.columns if pd.api.types.is_numeric_dtype(sheet2_cleaned[col])]
    corr_matrix = sheet2_cleaned[numeric_cols].corr()
<img width="692" height="515" alt="image" src="https://github.com/user-attachments/assets/7dd4d203-8fdd-42ab-a86b-b705c19d75e1" />
#Correlation Heatmap

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title("Feature Correlation Matrix")
    plt.show()
<img width="772" height="580" alt="image" src="https://github.com/user-attachments/assets/2297fce5-f311-4fb9-b193-f24886f66b85" />

#Dimensionality Reduction

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sheet2_cleaned[numeric_cols])
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=sheet2_cleaned['NSP'])
    plt.title("PCA Projection of CTG Data")
    plt.show()

#MIN_MAX SCALER

    X = sheet2_cleaned[numeric_cols]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = sheet2_cleaned['NSP']
    
    selector = SelectKBest(score_func=chi2, k=10)
    selector.fit(X_scaled, y)
    
    selected_features = [numeric_cols[i] for i in selector.get_support(indices=True)]
    print("Top 10 features:", selected_features)

#Visual Inspection with BoxPlots 

    plt.figure(figsize=(16, 10))
    sns.boxplot(data=X)
    plt.xticks(rotation=90)
    plt.title("Boxplot of Numeric Features")
    plt.show()

<img width="799" height="539" alt="image" src="https://github.com/user-attachments/assets/0c14169e-31a1-4904-b31a-391ac1397536" />

# Compute z-scores for FM
    sheet2_cleaned['FM_zscore'] = zscore(sheet2_cleaned['FM'])

# Flag anomalies: z-score > 3 or < -3
    sheet2_cleaned['FM_anomaly'] = sheet2_cleaned['FM_zscore'].abs() > 3

# Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sheet2_cleaned,
        x='FM',
        y='NSP',
        hue='FM_anomaly',
        palette={True: 'red', False: 'blue'},
        alpha=0.7
    )
    
    plt.title('FM vs NSP with Anomalies Highlighted')
    plt.xlabel('FM (Fetal Movement)')
    plt.ylabel('NSP (Fetal State)')
    plt.legend(title='FM Anomaly')        
<img width="784" height="505" alt="image" src="https://github.com/user-attachments/assets/df5c2ccc-6a1f-4f74-86ae-38665bd39241" />

# Compute z-scores for FM
    sheet2_cleaned['FM_zscore'] = zscore(sheet2_cleaned['FM'])

# Flag anomalies: z-score > 3 or < -3
    sheet2_cleaned['FM_anomaly'] = sheet2_cleaned['FM_zscore'].abs() > 3

# Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sheet2_cleaned,
        x='FM',
        y='NSP',
        hue='FM_anomaly',
        palette={True: 'red', False: 'blue'},
        alpha=0.7
    )
    
    plt.title('FM vs NSP with Anomalies Highlighted')
    plt.xlabel('FM (Fetal Movement)')
    plt.ylabel('NSP (Fetal State)')
    plt.legend(title='FM Anomaly')
    plt.tight_layout()
    plt.show()
<img width="774" height="469" alt="image" src="https://github.com/user-attachments/assets/bf7bf894-011e-4c3d-aa42-be21c1bba4bf" />


#Zscore for all features

     z_scores = np.abs(zscore(X))
    anomaly_mask = (z_scores > 3)
    anomaly_counts = anomaly_mask.sum(axis=0)
    
    pd.DataFrame({
        'Feature': X.columns,
        'Anomalies': anomaly_counts
    }).sort_values(by='Anomalies', ascending=False)

<img width="245" height="731" alt="image" src="https://github.com/user-attachments/assets/c678b157-b842-467c-9ed1-17183a249145" />

  
#AC VS ALTV

    sns.scatterplot(data=sheet2_cleaned, x='ALTV', y='AC', hue='NSP', palette='Set1')
    plt.title('AC vs ALTV Colored by NSP Class')
    plt.xlabel('ALTV')
    plt.ylabel('AC')
    plt.show()

<img width="737" height="477" alt="image" src="https://github.com/user-attachments/assets/6ebebcb3-1737-4092-9252-0325f3007474" />

#AC VS DP

    sns.scatterplot(data=sheet2_cleaned, x='ALTV', y='DP', hue='NSP', palette='Set1')
    plt.title('ATLV VS DP Colored by NSP Class')
    plt.xlabel('ALTV')
    plt.ylabel('DP')
    plt.show()

<img width="738" height="490" alt="image" src="https://github.com/user-attachments/assets/c12bc2bf-4e39-43ac-8f19-9aa92fe97b7f" />

# Compute z-scores for ALTV
    sheet2_cleaned['ALTV_z'] = zscore(sheet2_cleaned['ALTV'])
# Flag anomalies: z-score > 3
    sheet2_cleaned['ALTV_anomaly'] = sheet2_cleaned['ALTV_z'].abs() > 3
# Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sheet2_cleaned,
        x='ALTV',
        y='NSP',
        hue='ALTV_anomaly',
        palette={True: 'red', False: 'blue'},
        alpha=0.7
    )
    plt.title('ALTV vs NSP with Anomalies Highlighted')
    plt.xlabel('ALTV (% Abnormal Long-Term Variability)')
    plt.ylabel('NSP (Fetal State)')
    plt.legend(title='ALTV Anomaly')
    plt.tight_layout()
    plt.show()
    
 <img width="788" height="465" alt="image" src="https://github.com/user-attachments/assets/c13aafb6-fc40-4622-83f6-ffd6435ca0bc" />

#Analyse mean, std of DATA
    
    sheet2_cleaned.describe().T

<img width="524" height="725" alt="image" src="https://github.com/user-attachments/assets/775db4a6-0e72-4156-a138-145ac8ecc5ad" />


