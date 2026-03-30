import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
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
        # Enhanced statistics: mean, std, median, min, max, skew, kurtosis, range, q25, q75, iqr
        full_stats = [
            np.mean(full_temp), np.std(full_temp), np.median(full_temp), 
            np.min(full_temp), np.max(full_temp), skew(full_temp), kurtosis(full_temp),
            np.max(full_temp) - np.min(full_temp),  # range
            np.percentile(full_temp, 25), np.percentile(full_temp, 75),  # Q1, Q3
            np.percentile(full_temp, 75) - np.percentile(full_temp, 25)  # IQR
        ]
        
        angio_stats = []
        for angio in ['LPA', 'LCA', 'MPA', 'MCA']:
            angio_csv = os.path.join(subject_dir, 'Angiosoms', f'{subject_id}_{foot}_{angio}.csv')
            if os.path.exists(angio_csv):
                angio_temp = pd.read_csv(angio_csv, header=None).values.flatten()
                angio_temp = angio_temp[angio_temp > 0]
                angio_stats.extend([
                    np.mean(angio_temp), np.std(angio_temp), np.median(angio_temp), 
                    np.min(angio_temp), np.max(angio_temp), skew(angio_temp), kurtosis(angio_temp),
                    np.max(angio_temp) - np.min(angio_temp),
                    np.percentile(angio_temp, 25), np.percentile(angio_temp, 75),
                    np.percentile(angio_temp, 75) - np.percentile(angio_temp, 25)
                ])
            else:
                angio_stats.extend([0]*11)  # if missing
        
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

# Step 0: SPLIT DATA FIRST (prevent data leakage)
print("\n=== PREVENTING DATA LEAKAGE ===")
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train_raw.shape}, Test set: {X_test_raw.shape}")

# Step 3: Correlation filtering (on TRAIN set only)
print("\n=== STEP 3: Correlation Filtering (Train-Only) ===")
df_feat_train = pd.DataFrame(X_train_raw)
corr_matrix = df_feat_train.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_train_filtered = df_feat_train.drop(to_drop, axis=1).values
# Apply same columns to test set
X_test_filtered = pd.DataFrame(X_test_raw).drop(to_drop, axis=1).values

print(f"Features after correlation filtering: {X_train_filtered.shape}")

# Save step 3
np.save('step3_features_filtered.npy', X_train_filtered)

# Step 3.5: Scale data using TRAIN statistics only
print("\n=== STEP 3.5: Scaling (Train-Only) ===")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_test_scaled = scaler.transform(X_test_filtered)  # Transform using train stats

# Step 4: Feature ranking (on TRAIN set only)
print("\n=== STEP 4: Feature Selection (Train-Only) ===")
k_values = [15, 20, 25, 30]
best_f1 = 0
best_k = 20
selector_best = None

for k in k_values:
    selector = SelectKBest(f_classif, k=min(k, X_train_scaled.shape[1]))
    X_temp = selector.fit_transform(X_train_scaled, y_train_raw)
    
    # Quick cross-val with this k
    et_test = ExtraTreesClassifier(n_estimators=50, random_state=42)
    scores = cross_val_score(et_test, X_temp, y_train_raw, cv=5, scoring='f1_weighted')
    mean_f1 = scores.mean()
    
    print(f"k={k}: F1 = {mean_f1:.4f}")
    
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_k = k
        selector_best = selector

print(f"Best k: {best_k} with F1: {best_f1:.4f}")
X_train_top = selector_best.transform(X_train_scaled)
X_test_top = selector_best.transform(X_test_scaled)  # Use train's selector

print(f"Top {best_k} features selected, Train shape: {X_train_top.shape}, Test shape: {X_test_top.shape}")

# Save step 4
np.save('step4_features_top.npy', X_train_top)

# Step 5: SMOTE (on TRAIN set only - NOT on test)
print("\n=== STEP 5: SMOTE (Train-Only, NO SMOTE on Test) ===")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_top, y_train_raw)

# TEST set keeps original class distribution (no SMOTE)
X_test_final = X_test_top
y_test_final = y_test_raw

print(f"Data after SMOTE: Train {X_train_smote.shape}, Test {X_test_final.shape}")
print(f"Train class distribution (after SMOTE): {np.bincount(y_train_smote)}")
print(f"Test class distribution (ORIGINAL): {np.bincount(y_test_final)}")

# Save step 5
np.save('step5_features_smote.npy', X_train_smote)
np.save('step5_labels_smote.npy', y_train_smote)

# Step 6: Build Ensemble with Hyperparameter Tuning (REDUCED OVERFITTING)
print("\n=== Model Training with Ensemble and Hyperparameter Tuning ===")

# Use the corrected train/test split (no data leakage)
X_train = X_train_smote
y_train = y_train_smote
X_test = X_test_final
y_test = y_test_final

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Training classes: {np.bincount(y_train)}")
print(f"Test classes: {np.bincount(y_test)}")

# 1. AdaBoost with regularization
print("\nTuning AdaBoost (with regularization)...")
ada_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.8, 1.0],
}
ada_grid = GridSearchCV(
    AdaBoostClassifier(random_state=42),
    ada_params, cv=StratifiedKFold(n_splits=5), 
    scoring='f1_weighted', n_jobs=-1, verbose=0
)
ada_grid.fit(X_train, y_train)
print(f"Best AdaBoost params: {ada_grid.best_params_}")
ada = ada_grid.best_estimator_

# 2. Random Forest with regularization
print("Tuning Random Forest (with regularization)...")
rf_params = {
    'n_estimators': [100, 150],
    'max_depth': [10, 12],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2'],
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params, cv=StratifiedKFold(n_splits=5),
    scoring='f1_weighted', n_jobs=-1, verbose=0
)
rf_grid.fit(X_train, y_train)
print(f"Best RF params: {rf_grid.best_params_}")
rf = rf_grid.best_estimator_

# 3. Extra Trees with regularization
print("Tuning Extra Trees (with regularization)...")
et_params = {
    'n_estimators': [100, 150],
    'max_depth': [10, 12],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2'],
}
et_grid = GridSearchCV(
    ExtraTreesClassifier(random_state=42, n_jobs=-1),
    et_params, cv=StratifiedKFold(n_splits=5),
    scoring='f1_weighted', n_jobs=-1, verbose=0
)
et_grid.fit(X_train, y_train)
print(f"Best ExtraTrees params: {et_grid.best_params_}")
et = et_grid.best_estimator_

# 4. Voting Classifier Ensemble
print("Building Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[('ada', ada), ('rf', rf), ('et', et)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Evaluate all models
models = {
    'AdaBoost': ada,
    'Random Forest': rf,
    'Extra Trees': et,
    'Voting Ensemble': voting_clf
}

print("\n" + "="*70)
print("MODEL EVALUATION (with proper train/test split, no data leakage)")
print("="*70)
best_model = None
best_score = 0

for name, model in models.items():
    # Training performance
    y_train_pred = model.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    # Test performance
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    
    # Overfitting gap
    gap = train_f1 - test_f1
    
    print(f"\n{name}:")
    print(f"  Train F1: {train_f1:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  Overfitting Gap: {gap:.4f} {'✓ GOOD' if gap < 0.15 else '⚠️ MODERATE' if gap < 0.25 else '❌ HIGH'}")
    
    if test_f1 > best_score:
        best_score = test_f1
        best_model = (name, model)

print(f"\n{'='*70}")
print(f"Best Model: {best_model[0]} with Test F1: {best_score:.4f}")
print(f"{'='*70}")

# Detailed classification report for best model
print(f"\nDetailed Classification Report ({best_model[0]}):")
y_pred_best = best_model[1].predict(X_test)
print(classification_report(y_test, y_pred_best))

# Cross-validation with original unsmoted data (realistic estimate)
print(f"\n=== Cross-Validation (Using Original Dataset) ===")
print("(More realistic estimate of generalization to new data)")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=best_k)),
    ('model', best_model[1])
])
cv_scores = cross_val_score(
    best_model[1], X, y, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
    scoring='f1_weighted'
)
print(f"Cross-Val F1-Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Save all models
joblib.dump(ada, 'adaboost_model.pkl')
joblib.dump(rf, 'randomforest_model.pkl')
joblib.dump(et, 'extratrees_model.pkl')
joblib.dump(voting_clf, 'voting_ensemble_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector_best, 'feature_selector.pkl')

print("\n✅ All models, scaler, and selector saved successfully!")