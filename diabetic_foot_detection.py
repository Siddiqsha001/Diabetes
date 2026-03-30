import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import skew, kurtosis

# Step 1: Load data
data_dir_control = r'd:\Documents\Projects\Manjari proj\datasets\Control Group-20260330T063744Z-1-001\Control Group'
data_dir_dm = r'd:\Documents\Projects\Manjari proj\datasets\DM Group-20260330T063807Z-1-001\DM Group'

feature_list = []
labels = []

def load_features(subject_dir, label):
    subject_id = os.path.basename(subject_dir)
    gender = 1 if '_M' in subject_id else 0  # 1 for male, 0 for female
    for foot in ['L', 'R']:
        full_csv = os.path.join(subject_dir, f'{subject_id}_{foot}.csv')
        if not os.path.exists(full_csv):
            continue
        full_temp = pd.read_csv(full_csv, header=None).values.flatten()
        full_temp = full_temp[full_temp > 0]  # remove zeros
        full_stats = [np.mean(full_temp), np.std(full_temp), np.median(full_temp), np.min(full_temp), np.max(full_temp)]
        
        angio_stats = []
        for angio in ['LPA', 'LCA', 'MPA', 'MCA']:
            angio_csv = os.path.join(subject_dir, 'Angiosoms', f'{subject_id}_{foot}_{angio}.csv')
            if os.path.exists(angio_csv):
                angio_temp = pd.read_csv(angio_csv, header=None).values.flatten()
                angio_temp = angio_temp[angio_temp > 0]
                angio_stats.extend([np.mean(angio_temp), np.std(angio_temp), np.median(angio_temp), np.min(angio_temp), np.max(angio_temp)])
            else:
                angio_stats.extend([0]*5)  # if missing
        
        feat = full_stats + angio_stats + [gender]
        feature_list.append(feat)
        labels.append(label)

# Load control
for subject in os.listdir(data_dir_control):
    subject_dir = os.path.join(data_dir_control, subject)
    if os.path.isdir(subject_dir):
        load_features(subject_dir, 0)

# Load DM
for subject in os.listdir(data_dir_dm):
    subject_dir = os.path.join(data_dir_dm, subject)
    if os.path.isdir(subject_dir):
        load_features(subject_dir, 1)

X = np.array(feature_list)
y = np.array(labels)

print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

# Save step 1 output
np.save('step1_features.npy', X)
np.save('step1_labels.npy', y)

# Step 2: Feature extraction (already done, but add more if needed)
# For NTR like, but since no ranges defined, skip for now

# Step 3: Correlation filtering
# Compute correlation matrix
df_feat = pd.DataFrame(X)
corr_matrix = df_feat.corr()
# Remove features with correlation > 0.95
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_filtered = df_feat.drop(to_drop, axis=1).values

print(f"Features after correlation filtering: {X_filtered.shape}")

# Save step 3
np.save('step3_features_filtered.npy', X_filtered)

# Step 4: Feature ranking
# Use Extra Trees for feature importance (as in the paper)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)
et.fit(X_filtered, y)
importances = et.feature_importances_
indices = np.argsort(importances)[::-1]

# Select top 20 features
top_k = 20
top_features = indices[:top_k]
X_top = X_filtered[:, top_features]

print(f"Top {top_k} features selected")

# Save step 4
np.save('step4_features_top.npy', X_top)

# Step 5: SMOTE
scaler = StandardScaler()
X_top_scaled = scaler.fit_transform(X_top)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_top_scaled, y)

print(f"Data after SMOTE: {X_smote.shape}, {y_smote.shape}")

# Save step 5
np.save('step5_features_smote.npy', X_smote)
np.save('step5_labels_smote.npy', y_smote)

# Step 6: AdaBoost
ada = AdaBoostClassifier(n_estimators=200, random_state=42)
scores = cross_val_score(ada, X_smote, y_smote, cv=5, scoring='f1')
print(f"Cross-validation F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Train and test
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(ada, 'adaboost_model.pkl')

print("Pipeline completed. Outputs saved.")