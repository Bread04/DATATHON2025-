# DATATHON2025-
The main objective of this challenge is to use real-world CTG recordings to build a solution that supports clinicians in spotting fetal distress. By analyzing patterns in the data, you are expected to design a system that can reliably separate Normal, Suspect, and Pathologic cases. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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
class_counts = sheet2_cleaned[target_col].value_counts().sort_index()
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
redundant_features = ['AC.1', 'FM.1', 'UC.1', 'DL.1', 'DS.1', 'DP.1']
symbolic_features = ['A', 'B', 'C', 'D', 'E', 'AD', 'DE', 'LD', 'FS', 'SUSP', 'CLASS']
target_variable = ['NSP']

# Organize into dictionary
feature_groups = {
    'Physiological Signals': physiological_signals,
    'Variability Metrics': variability_metrics,
    'Histogram Descriptors': histogram_descriptors,
    'Timing Markers': timing_markers,
    'Baseline References': baseline_references,
    'Redundant Features': redundant_features,
    'Symbolic Features': symbolic_features,
    'Target Variable': target_variable
}

# Display groupings
for group, cols in feature_groups.items():
    print(f"\n🧠 {group} ({len(cols)} features):")
    for col in cols:
        print(f"• {col}")
