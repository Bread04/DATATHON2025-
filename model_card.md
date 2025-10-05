# 🧠 Model Card: CTGClassifier (PyTorch)

## Model Overview
The `CTGClassifier` is a feedforward neural network trained to classify fetal states (NSP: Normal, Suspect, Pathologic) using cleaned CTG signals. It uses 10 statistically selected features and is trained with class-balanced sampling to mitigate bias toward majority classes.

- **Architecture**: 2-layer MLP with ReLU activation
- **Input Features**: Top 10 features selected via chi-squared scoring
- **Output Classes**: 3 (NSP = 1, 2, 3)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Training Epochs**: 100
- **Class Imbalance Handling**: Stratified split + optional class weights

## Intended Use
This model is designed for educational and diagnostic prototyping. It supports clinicians and researchers in identifying fetal distress patterns from CTG recordings. It is not intended for direct clinical deployment without further validation.

## Ethical Safeguards
- **Label Leakage Prevention**: Symbolic and diagnosis-encoding features (e.g., `CLASS`, `SUSP`) were excluded from training.
- **Class Imbalance**: Rare but critical cases (NSP = 3) were preserved and weighted to ensure fair representation.
- **Threshold Logic (Optional)**: A custom threshold can be applied to increase confidence before labeling a case as pathologic:
  ```python
