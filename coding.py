import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score
import joblib

# 1. Load Excel file
df = pd.read_excel('ctg_data.xlsx')  # Update with your actual file path
feature_cols = ['AC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'Min', 'Variance']
target_col = 'NSP'

df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(int)

# 2. Split data
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 3. Define pipeline
nn_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('nn', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', max_iter=500, random_state=42))
])

# 4. Train model
nn_pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = nn_pipeline.predict(X_test)
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save model
joblib.dump(nn_pipeline, 'ctg_nn_model.pkl')

