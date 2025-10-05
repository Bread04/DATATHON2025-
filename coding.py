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

ctg_path = Path()  # Update if needed
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

#Clean headers and data
def clean_ctg_sheet(df):

    # Drop empty columns and rows
    df = df.dropna(axis=1, how='all').dropna()

    # Reset index
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
           

# Define column groups
physiological_signals = ['AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR']
variability_metrics = ['ASTV', 'mSTV', 'ALTV', 'mLTV']
histogram_descriptors = ['Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
timing_markers = ['b', 'e']
baseline_references = ['LB', 'LBE']
redundant_features = ['b', 'e','AC.1', 'FM.1', 'UC.1', 'DL.1', 'DS.1', 'DP.1','CLASS','A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP']
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

# Step 3: Select Features Based on Visual Insights
selected_features = ['AC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'Min', 'Variance']
X = sheet2_cleaned[selected_features]

#Step 4 : Add Ethical Alert Logic
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

y_pred_custom = np.array([
    3 if p[2] > 0.3 else np.argmax(p) + 1
    for p in probs
])

print("Custom Threshold — NSP = 3 if prob > 0.3")


import pandas as pd

feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importance.sort_values().plot(kind='barh', title='Feature Importance')
plt.show()


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



