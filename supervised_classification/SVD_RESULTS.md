# SVD and handcrafted classification comparison

**All 107,180 image IDs, labels and split memberships match the supplied SVD CSV exactly.**

80,000 train; 20,000 validation; 7,180 independent test images excluded from training and scoring.

| model | features | n_features | accuracy | macro_f1 | weighted_f1 | fit_seconds | convergence_warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | All_handcrafted | 162 | 0.9639 | 0.96317 | 0.9639 | 36.65973 | False |
| LogisticRegression | All_handcrafted | 162 | 0.94775 | 0.94701 | 0.94764 | 30.03731 | False |
| RandomForest | RGB | 48 | 0.93515 | 0.93316 | 0.93505 | 14.33292 | False |
| LogisticRegression | RGB_LBP | 102 | 0.9229 | 0.92079 | 0.92262 | 11.67415 | False |
| LogisticRegression | Texture | 114 | 0.904 | 0.90312 | 0.90393 | 16.89709 | False |
| LogisticRegression | RGB_GLCM | 84 | 0.90025 | 0.89919 | 0.89989 | 10.48756 | False |
| DecisionTree | All_handcrafted | 162 | 0.8995 | 0.89768 | 0.89941 | 29.98938 | False |
| RandomForest | LBP | 54 | 0.886 | 0.88567 | 0.88533 | 19.79911 | False |
| RandomForest | GLCM | 36 | 0.83075 | 0.82849 | 0.82982 | 22.67582 | False |
| RandomForest | Gabor | 24 | 0.83085 | 0.82746 | 0.82981 | 16.52751 | False |
| LogisticRegression | LBP | 54 | 0.82505 | 0.82239 | 0.82415 | 12.15332 | False |
| LogisticRegression | RGB | 48 | 0.78555 | 0.78145 | 0.78313 | 9.54167 | False |
| LogisticRegression | GLCM | 36 | 0.7445 | 0.73753 | 0.74152 | 11.24643 | False |
| RandomForest | SVD_32 | 32 | 0.70775 | 0.69752 | 0.70252 | 20.46895 | False |
| RandomForest | SVD_64 | 64 | 0.6943 | 0.68279 | 0.6882 | 34.25451 | False |
| RandomForest | SVD_128 | 128 | 0.67505 | 0.66032 | 0.66608 | 45.96578 | False |
| RandomForest | SVD_256 | 256 | 0.66845 | 0.65038 | 0.65649 | 65.39566 | False |
| LogisticRegression | Gabor | 24 | 0.65205 | 0.64242 | 0.64403 | 5.46598 | False |
| LogisticRegression | SVD_32 | 32 | 0.5416 | 0.50171 | 0.51096 | 1.2926 | False |
| LogisticRegression | SVD_128 | 128 | 0.53685 | 0.49825 | 0.5073 | 1.72403 | False |
| LogisticRegression | SVD_64 | 64 | 0.53925 | 0.49787 | 0.50741 | 1.37468 | False |
| LogisticRegression | SVD_256 | 256 | 0.5321 | 0.49693 | 0.50517 | 2.7561 | False |
| Dummy | RGB | 48 | 0.1432 | 0.02784 | 0.03588 | 0.08637 | False |

Best SVD: RandomForest, SVD_32, macro F1 0.6975, accuracy 0.7077.

## Interpretation
- On this shared validation split, handcrafted features outperform the evaluated pixel-SVD representations for both Logistic Regression and Random Forest.
- A large explained-variance ratio is not the same as strong class prediction: SVD preserves pixel variation without using labels.
- Nonlinear Random Forest improves over Logistic Regression on SVD here. This does not prove a particular causal explanation for the gap.
- These are internal validation comparisons; no independent-test score is reported. No hyperparameter search was performed.
- Explicit brightness columns remain unavailable; these experiments do not claim a separate brightness feature evaluation.
- SVD preprocessing was supplied by Person 3. This run verified feature integrity and split alignment, not a fresh audit/refit of the original pixel transformation.

## Reproduce SVD-only experiments
```bash
python train_svd.py --svd svd_32px_features.csv --manifest split_manifest.csv --out results
```
The large input CSVs are not bundled. Obtain them from the team.