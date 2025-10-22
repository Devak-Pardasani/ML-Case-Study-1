# Devak Pardasani - Final Stacking + IterativeImputer + Custom Scorer

import numpy as np
import pandas as pd

# Enable IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# ---- Custom metrics ----
def tpr_at_fpr_score(y_true, y_proba, target_fpr=0.01, **kwargs):
    fpr, tpr, _ = roc_curve(y_true, y_proba, drop_intermediate=False)
    return float(np.interp(target_fpr, fpr, tpr))

def combined(y_true, y_proba, w_tpr=0.1, w_auc=0.9, **kwargs):
    return w_auc * roc_auc_score(y_true, y_proba) + w_tpr * tpr_at_fpr_score(y_true, y_proba)

# ---- Data loading ----
def gen_train():
    train = pd.read_csv("spamTrain1.csv", header=None)
    X_train_raw, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    return X_train_raw, y_train

def gen_test():
    test = pd.read_csv("spamTrain2.csv", header=None)
    X_test_raw, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    return X_test_raw, y_test

# ---- Main pipeline ----
def main():
    # Scorers
    tpr1_scorer = make_scorer(tpr_at_fpr_score, needs_proba=True)
    combo_score = make_scorer(combined, needs_proba=True)

    # Load data
    X_train_raw, y_train = gen_train()
    X_test_raw, y_test = gen_test()

    # ---- Feature selection (example: all features) ----
    selected_features = [i for i in range(X_train_raw.shape[1])]
    X_train_raw = X_train_raw.iloc[:, selected_features]
    X_test_raw = X_test_raw.iloc[:, selected_features]

    # ---- Step 1: Pre-tune XGB alone ----
    pipe_xgb = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=42)),
        ("clf", XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=1.415,
            objective="binary:logistic"
        ))
    ])

    param_grid_xgb = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [3, 5],
        "clf__learning_rate": [0.05, 0.1]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_xgb = GridSearchCV(
        estimator=pipe_xgb,
        param_grid=param_grid_xgb,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_xgb.fit(X_train_raw, y_train)
    best_xgb = grid_xgb.best_estimator_.named_steps["clf"]

    print("Best XGB params:", grid_xgb.best_params_)
    print(f"Best XGB CV AUC: {grid_xgb.best_score_:.4f}")

    # ---- Step 2: Stacking classifier with pre-tuned XGB ----
    stack = StackingClassifier(
        estimators=[
            ("xgb", best_xgb),
            ("lr", LogisticRegression(solver="liblinear"))
        ],
        final_estimator=LogisticRegression(solver="liblinear"),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1
    )

    pipe_stack = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=42)),
        ("stack", stack)
    ])

    # Fit stacking classifier
    pipe_stack.fit(X_train_raw, y_train)

    # ---- Evaluate on test set ----
    p_test = pipe_stack.predict_proba(X_test_raw)[:, 1]
    print(f"TEST AUC: {roc_auc_score(y_test, p_test):.4f}")
    print(f"TEST TPR@1%: {tpr_at_fpr_score(y_test, p_test, 0.01):.4f}")

    # ---- Permutation importance (XGB only) ----
    print("\nComputing permutation feature importance using XGB base model...")
    xgb_clf = best_xgb
    result = permutation_importance(
        xgb_clf, X_test_raw, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    feature_names = [f"feat_{i}" for i in range(X_test_raw.shape[1])]
    perm_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": result.importances_mean,
        "Std": result.importances_std
    }).sort_values("Importance", ascending=False)

    print("\nTop 10 most important features (permutation):")
    print(perm_df.head(10))

    # Plot top 20
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=perm_df.head(20),
        x="Importance",
        y="Feature",
        palette="viridis"
    )
    plt.title("Top 20 Feature Importances (Permutation Importance)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
