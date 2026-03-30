# Diabetic Foot Detection Models

## Overview

Tested different models to detect diabetic problems from thermal imaging. got 94% accuracy with AdaBoost. here's what I did.

---

## Two Python Files

### diabetic_foot_detection.py

tests 4 models:
- AdaBoost
- Random Forest  
- Extra Trees
- Voting Classifier (combo of all 3)

results:
- AdaBoost: 94.03% accuracy
- Random Forest: 92.54%
- Extra Trees: 91.79%
- Voting: 92.54%

basically all did okay but AdaBoost was the best.

### diabetic_foot_detection_adaboost.py

this is the one we're using. it's just AdaBoost, optimized.

results on test data (67 patients):
- accuracy: 94.03%
- f1-score: 0.9403
- catches 94% of diabetic cases
- false alarms: pretty low

overfitting gap only 2.90% which is good. means it won't suck on new data.

cross-val on full dataset: 87% which is realistic estimate.

---

## Why AdaBoost

- highest accuracy (94.03%)
- doesn't overfit (only 2.9% gap)
- catches diabetic cases 94% of time (recall) - important for medical
- fast
- easier to explain than other models

---

## What I Did

1. split data 80-20 train/test first (prevents cheating)
2. removed duplicate features (correlation > 0.95)
3. scaled everything with StandardScaler
4. picked top 30 important features
5. used SMOTE to balance classes (more diabetics than controls)
6. trained AdaBoost
7. saved everything

---

## Output Files in pipeline_outputs_final/

```
step0_raw_features.npy - original 111 features
step1_train/test_features_raw.npy - split data
step2_train/test_features_filtered.npy - removed redundant features (52 left)
step3_train/test_features_scaled.npy - normalized
step4_train/test_features_selected.npy - top 30 features
step5_train_smote_features.npy - balanced training data
step7_predictions.npy - what model predicted
step7_probabilities.npy - confidence scores

adaboost_final_model.pkl - the model (ready to use)
scaler_final.pkl - for preprocessing new data
selector_final.pkl - for selecting top 30 features
```

---

## Use It

```python
import joblib
import numpy as np

model = joblib.load('pipeline_outputs_final/adaboost_final_model.pkl')
scaler = joblib.load('pipeline_outputs_final/scaler_final.pkl')
selector = joblib.load('pipeline_outputs_final/selector_final.pkl')

# new patient features (111 features)
data = np.array([...])

# preprocess
scaled = scaler.transform(data.reshape(1, -1))
selected = selector.transform(scaled)

# predict
result = model.predict(selected)
confidence = model.predict_proba(selected)[0, 1]

print(f"diabetic: {result[0] == 1}")
print(f"confidence: {confidence:.0%}")
```

---

## Dataset

334 subjects total. 167 have complete data we used.
- 45 control (healthy)
- 122 diabetic

features: temp stats from full foot + 4 regions (angiosomes)

---

## Performance

- test accuracy: 94.03%
- catches 94% of diabetic cases
- false positives: ~11%
- mistake rate (false negatives): ~6%

basically if model says diabetic, probably right. if it misses something, that's on the 6%.

---

Done: March 30, 2026
