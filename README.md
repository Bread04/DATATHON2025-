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

From the data seen, it is obvious that the weight distribution for NSP 2 and 3 classes is uneven;
    Class 1 (Normal) forms a dense cluster
    Classes 2 and 3 are more dispersed
    Some points lie far from all clusters → potential anomalies 

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y)
    plt.title("PCA Projection of CTG Data")
    plt.show()
    
    # Your PCA scatter plot shows:
    #Class 1 (Normal) forms a dense cluster
    #Classes 2 and 3 are more dispersed
    #Some points lie far from all clusters → potential anomalies
<img width="597" height="435" alt="image" src="https://github.com/user-attachments/assets/a6cddbf7-e463-4aa0-8536-f22a89691afb" />

Ensures that rare classes (e.g., NSP = 3) are not ignored by the model
Helps classifiers like Logistic Regression, Random Forest, or XGBoost treat all classes fairly
Prevents bias toward majority class (NSP = 1)

    # 1. Inspect the target column
    sheet2_cleaned[target_variable].value_counts(dropna=False)
    sheet2_cleaned[target_variable].unique()
    
    # 2. Drop or impute missing labels
    y = sheet2_cleaned[target_variable]
    
    # Option A – drop rows with missing NSP values
    mask = y.notna()
    y = y.loc[mask]
    
    # Optionally keep the aligned feature matrix if needed later
    # X = clean_df.loc[mask, feature_cols]
    
    # 3. Make sure the dtype is consistent (integers)
    y = y.astype(int)
    
    # 4. Recompute classes and weights
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced',
                                   classes=classes,
                                   y=y)
    
    class_weight_dict = dict(zip(classes, weights))
    class_weight_dict

Model Used to Evaluate the AI MODELS (SHARED)

    def evaluate_model(name, estimator, X_train, y_train, X_test, y_test, fit_kwargs=None, display_report=False):
        fit_kwargs = fit_kwargs or {}
        estimator.fit(X_train, y_train, **fit_kwargs)
        y_pred = estimator.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append({'Model': name, 'Balanced Accuracy': bal_acc, 'F1 Macro': f1})
        print(f"{name} — Balanced Accuracy: {bal_acc:.3f}, Macro F1: {f1:.3f}")
        if display_report:
            print(classification_report(y_test, y_pred, digits=3))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=sorted(classes), normalize='true', cmap='Blues'
        )
        disp.ax_.set_title(f"{name} — Normalized Confusion Matrix")
        plt.show()
        return estimator, y_pred

    classes

#DATA MODELLING INTO THE ACTUAL ML MACHINE

    from pathlib import Path
    import warnings
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier






# Inspect and Clean Target Label
    target_variable = 'NSP'
    y = sheet2_cleaned[target_variable]
    mask = y.notna()
    y = y.loc[mask].astype(int)
    
## Step 2: Compute Class Weights for Imbalance
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    
# Step 3: Select Features Based on Visual Insights (Most Prominent)
    selected_features = ['AC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'Min', 'Variance']
    X = sheet2_cleaned[selected_features]
    
 # Step 4 : Add Ethical Alert Logic
    
    def ethical_alert(row):
        if row['DP'] > 0 and row['ALTV'] > 80:
            return 'Review: Possible distress'
        elif row['AC'] < 5 and row['ASTV'] < 10:
            return 'Review: Low reactivity'
        else:
            return 'No alert'
    
    sheet2_cleaned['Ethical_Alert'] = sheet2_cleaned.apply(ethical_alert, axis=1)
    
# Step 5: Train/Test Split
    from sklearn.model_selection import train_test_split
    
    y = sheet2_cleaned['NSP'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
# Indicate Threshold for NSP Class 3 because we want to ensure that the model can predict such cases super accurately, having super-high confidence before making a guess 
    y_pred_custom = np.array([
        3 if p[2] > 0.3 else np.argmax(p) + 1
        for p in probs
    ])
    
    print("Custom Threshold — NSP = 3 if prob > 0.3")
    
    
    import pandas as pd
    
    feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    feature_importance.sort_values().plot(kind='barh', title='Feature Importance')
    plt.show()
    
<img width="744" height="500" alt="image" src="https://github.com/user-attachments/assets/7066cac5-a19d-4138-9739-e38117473b94" />
  
# We show the features that are of most importance which affects NSP



#DECSION TREE MODEL

    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'class_weight': [class_weight_dict]
    }
    
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20, 10))
    plot_tree(best_tree, filled=True, feature_names=X.columns, class_names=['Normal', 'Suspect', 'Pathologic'])
    plt.title("Tuned Decision Tree — CTG Classification")
    plt.show()
    
    probs = best_tree.predict_proba(X_test)
    y_pred_custom = np.array([
        3 if p[2] > 0.3 else np.argmax(p) + 1
        for p in probs
    ])
    
    flagged = sheet2_cleaned.loc[
        (sheet2_cleaned['NSP'] == 1) & 
        (sheet2_cleaned['Ethical_Alert'] != 'No alert')
    ]
    _ = evaluate_model(
        'Gradient Boosting',
        gb_pipeline,
        X_train, y_train,
        X_test, y_test,
        fit_kwargs=gb_fit_kwargs,
        display_report=True
    )
<img width="861" height="1107" alt="image" src="https://github.com/user-attachments/assets/c804334a-6900-4278-8a9c-055798fcec48" />
#GRADIENT BOOST MODEL

    from sklearn.ensemble import GradientBoostingClassifier
    
    gb_model = GradientBoostingClassifier(
        n_estimators=300,       # More trees for better learning
        learning_rate=0.05,     # Shrinks each tree’s contribution
        max_depth=3,            # Shallow trees reduce overfitting
        random_state=42
    )
    
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    gb_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('gb', gb_model)
    ])
    
    sample_weight_train = y_train.map(class_weight_dict)
    
    gb_fit_kwargs = {'gb__sample_weight': sample_weight_train}
    
    _ = evaluate_model(
        'Gradient Boosting',
        gb_pipeline,
        X_train, y_train,
        X_test, y_test,
        fit_kwargs=gb_fit_kwargs,
        display_report=True
    )
<img width="692" height="713" alt="image" src="https://github.com/user-attachments/assets/1b049fdd-6667-4433-acee-2a6cbde75003" />

Support Vector Machine

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    
    svm_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=2.0,
            gamma='scale',
            class_weight='balanced',
            probability=True  # Enables predict_proba for threshold tuning
        ))
    ])
    _ = evaluate_model(
        'Support Vector Machine',
        svm_pipeline,
        X_train, y_train,
        X_test, y_test,
        display_report=True
    )
    
   <img width="659" height="704" alt="image" src="https://github.com/user-attachments/assets/a03a4d68-587f-4ad2-9fc7-b0e675cc8a85" />

K-nearest neighbours

    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    
    knn_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=15))
    ])
    
    # Evaluate and capture metrics
    knn_metrics = evaluate_model(
        'k-Nearest Neighbors',
        knn_pipeline,
        X_train, y_train,
        X_test, y_test,
        display_report=True,
         # Ensure your function returns a dict of metrics
    )
<img width="719" height="706" alt="image" src="https://github.com/user-attachments/assets/44bf3277-195b-4f14-81c4-0af63cca5c8e" />
#Neural Network

        from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    nn_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('nn', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        ))
    ])
    
    _ = evaluate_model(
        'Neural Network',
        nn_pipeline,
        X_train, y_train,
        X_test, y_test,
        display_report=True
    )
<img width="786" height="709" alt="image" src="https://github.com/user-attachments/assets/67015494-0865-497b-aa0b-1f36a47f117a" />
    
# Aggregate model performance
    results_df = pd.DataFrame(results).sort_values('Balanced Accuracy', ascending=False)
    unique_models_df = results_df.drop_duplicates(subset='Model', keep='first')
    unique_models_df = unique_models_df.reset_index(drop=True)
    unique_models_df
    print(unique_models_df)
    
#Visualize Model

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.barh(unique_models_df['Model'], unique_models_df['Balanced Accuracy'], color='steelblue')
    plt.xlabel('Balanced Accuracy')
    plt.title('Best Version of Each Model')
    plt.gca().invert_yaxis()
<img width="806" height="600" alt="image" src="https://github.com/user-attachments/assets/5f0ce8f6-17d0-4a7b-a3f3-6d2b5696eab7" />

    plt.show()
