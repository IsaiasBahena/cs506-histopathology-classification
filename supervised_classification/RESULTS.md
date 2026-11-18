# Person 4 — November check-in results

**Internal validation. All image IDs, labels and split memberships now verified against the supplied SVD feature table.**

80,000 training images; 20,000 validation images; 7,180 independent test images excluded from fitting and scoring. Nine-class tissue-patch classification.

Best: RandomForest__All_handcrafted; accuracy 0.9639; macro F1 0.9632.
RGB + Logistic Regression baseline macro F1: 0.7815. Improvement: 18.17 percentage points; exceeds the proposed five-point target on this shared validation split.

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
| LogisticRegression | Gabor | 24 | 0.65205 | 0.64242 | 0.64403 | 5.46598 | False |
| Dummy | RGB | 48 | 0.1432 | 0.02784 | 0.03588 | 0.08637 | False |

## Best-model per-class metrics
| index | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| ADI | 0.99567 | 0.99472 | 0.99519 | 2082.0 |
| BACK | 0.99576 | 1.0 | 0.99787 | 2113.0 |
| DEB | 0.96982 | 0.96308 | 0.96643 | 2302.0 |
| LYM | 0.9974 | 0.99524 | 0.99632 | 2311.0 |
| MUC | 0.94225 | 0.95391 | 0.94804 | 1779.0 |
| MUS | 0.95958 | 0.95604 | 0.95781 | 2707.0 |
| NORM | 0.93192 | 0.92926 | 0.93059 | 1753.0 |
| STR | 0.92356 | 0.91958 | 0.92156 | 2089.0 |
| TUM | 0.95208 | 0.9574 | 0.95474 | 2864.0 |

## Most frequent errors (true → predicted)
- MUS → STR: 92 images.
- STR → MUS: 68 images.
- TUM → NORM: 59 images.
- NORM → TUM: 56 images.
- NORM → MUC: 50 images.
- DEB → TUM: 40 images.

## Evaluation rules and limits
- Image-level stratified 80/20 split, seed 4266457447. Development rows follow the sorted class/filename ordering used by the repository metadata builder. All 107,180 image memberships and labels were subsequently verified against the provided SVD feature CSV; no mismatches.
- StandardScaler was fitted only on training data for Logistic Regression. Trees used unscaled features.
- No hyperparameter tuning was performed. Logistic Regression uses C=1, lbfgs, max_iter=2000; Decision Tree uses defaults; Random Forest uses 100 trees. All seeds and feature columns are in source/config. No convergence warnings were recorded.
- Macro F1 weights the nine classes equally; accuracy weights individual images.
- SVD classification is now complete: see SVD_RESULTS.md and all_model_comparison.csv. Separate brightness features remain unavailable. RBF SVM and feature selection remain extensions.
- No patient/slide-level independence is established by the internal split. Do not describe these internal results as final generalization performance.
- Validation was used to compare and select models. Freeze settings before the one-time independent test.
- STR is cancer-associated stroma, not a malignant-cell label. This is not patient-level cancer diagnosis.