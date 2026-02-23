# Heart Disease Prediction — Learning Plan
**Competition:** Kaggle Playground Series S6E2
**Goal:** Binary classification — predict Presence/Absence of Heart Disease
**Metric:** ROC-AUC
**Data:** 630K train rows, 270K test rows, 13 features

---

## First Principles Philosophy
Each phase follows the pattern:
1. **Understand WHY** the technique works (math/intuition)
2. **Implement it** step by step
3. **Interpret the results** — what is the model telling us?
4. **Improve** based on what we learned

---

## Phase 1 — Exploratory Data Analysis (EDA)
**Notebook:** `heart-disease.ipynb`
**Goal:** Understand the data before touching any model.

### Steps
- [ ] Load data, inspect shape, dtypes, missing values
- [ ] Understand each feature medically (what does "Thallium" or "ST depression" mean?)
- [ ] Target distribution — is the dataset balanced?
- [ ] Univariate analysis — distributions of each feature
- [ ] Bivariate analysis — how does each feature relate to the target?
- [ ] Correlation matrix — which features are correlated with each other?
- [ ] Key questions to answer before modelling

**Key Learning:** Data tells you more than models do. EDA shapes every decision downstream.

---

## Phase 2 — Baseline Model (Logistic Regression)
**Goal:** Build the simplest possible model. Establish a score to beat.

### Steps
- [ ] Encode target (Presence=1, Absence=0)
- [ ] Train/validation split + 5-fold cross-validation (understand WHY we use CV)
- [ ] Understand the math: sigmoid function, log-loss, decision boundary
- [ ] Train Logistic Regression, evaluate ROC-AUC
- [ ] Confusion matrix, precision/recall tradeoff
- [ ] Interpret coefficients: which features push prediction toward disease?
- [ ] Establish baseline score

**Key Learning:** Logistic regression is a linear model — it draws a straight line in feature space. Understanding its limits tells us when to use something more powerful.

---

## Phase 3 — Tree-Based Models (Decision Tree → Random Forest)
**Goal:** Move beyond linear boundaries. Understand ensemble thinking.

### Steps
- [ ] Decision Tree: understand splits, gini/entropy, overfitting via depth
- [ ] Visualize a small tree — see exactly what the model learned
- [ ] Random Forest: why averaging many trees reduces variance (bias-variance tradeoff)
- [ ] Feature importances — which features does the forest rely on?
- [ ] Compare OOF (out-of-fold) ROC-AUC vs Logistic Regression

**Key Learning:** A single decision tree overfits. A forest of random trees generalizes. This is the core idea behind all ensemble methods.

---

## Phase 4 — Gradient Boosting (LightGBM)
**Goal:** Best-performing single model on tabular data. Understand boosting.

### Steps
- [ ] Understand boosting: each tree corrects residuals of the previous
- [ ] Key hyperparameters: learning_rate, num_leaves, max_depth, n_estimators
- [ ] Train with early stopping on validation set
- [ ] Feature importance (gain vs split)
- [ ] Tune key hyperparameters — understand what each one controls
- [ ] Compare vs Random Forest

**Key Learning:** Boosting builds trees sequentially, each one a "correction" to the last. Lower learning rate + more trees = better generalization (but slower).

---

## Phase 5 — Ensemble & Submit
**Goal:** Combine models for a more robust prediction. Submit to Kaggle.

### Steps
- [ ] Blend predictions: simple average of Logistic Regression + Random Forest + LightGBM
- [ ] Understand why blending works (models make different errors)
- [ ] Format submission CSV correctly
- [ ] Submit via Kaggle CLI or manually
- [ ] Record public leaderboard score
- [ ] Reflect: where is the gap between CV score and LB score?

**Key Learning:** No single model is perfect. Ensembling diverse models almost always improves score.

---

## Feature Reference
| Feature | Type | Medical Meaning |
|---|---|---|
| Age | Continuous | Patient age in years |
| Sex | Binary | 1 = Male, 0 = Female |
| Chest pain type | Categorical | 1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic |
| BP | Continuous | Resting blood pressure (mmHg) |
| Cholesterol | Continuous | Serum cholesterol (mg/dl) |
| FBS over 120 | Binary | Fasting blood sugar > 120 mg/dl (1=true) |
| EKG results | Categorical | 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy |
| Max HR | Continuous | Maximum heart rate achieved |
| Exercise angina | Binary | Exercise-induced angina (1=yes) |
| ST depression | Continuous | ST depression induced by exercise relative to rest |
| Slope of ST | Categorical | 1=upsloping, 2=flat, 3=downsloping |
| Number of vessels fluro | Continuous | Number of major vessels colored by fluoroscopy (0-3) |
| Thallium | Categorical | 3=normal, 6=fixed defect, 7=reversible defect |

---

## Progress Tracker
| Phase | Status | CV ROC-AUC | Notes |
|---|---|---|---|
| 1 - EDA | [ ] | — | — |
| 2 - Logistic Regression | [ ] | — | — |
| 3 - Random Forest | [ ] | — | — |
| 4 - LightGBM | [ ] | — | — |
| 5 - Ensemble + Submit | [ ] | — | — |
