# SDG-3 Heart-Disease Prediction 🫀

**Goal**  
Predict presence of heart disease (binary) using the UCI Heart-Disease dataset, supporting UN SDG-3 “Good Health and Well-being”.

**Dataset**  
303 patients, 14 clinical features.  
Source: `storage.googleapis.com/download.tensorflow.org/data/heart.csv`

**Model**  
Logistic Regression (baseline) – 80/20 train-test split.

**Results**  
Accuracy: **0.88**
Confusion matrix: see `screenshots/confusion_matrix.png`

**How to run**  
1. Open `heart_disease_model.py` in Google Colab.  
2. Run all cells.  
3. Charts auto-save to Colab file pane.

**Ethics note**  
Model is a teaching demo; real clinical use requires further validation & bias checks.

**Screenshots**  
dataset_preview.png  
model_accuracy.png  
confusion_matrix.png
![Alt confusion_matrix image](review/confusion_matrix.png "confusion matrix")