#Devak Pardasani - October 16th, 2025 - Case Study 1

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np, pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer

# ---- metric: TPR at exactly 1% FPR (interpolated) ----
def tpr_at_fpr_score(y_true, y_proba, target_fpr=0.01, **kwargs):
    fpr, tpr, _ = roc_curve(y_true, y_proba, drop_intermediate=False)
    return float(np.interp(target_fpr, fpr, tpr))


def main():
    tpr1_scorer = make_scorer(tpr_at_fpr_score, needs_proba=True)
    #Train
    X_train_raw, y_train = gen_train()
    #Test
    X_test_raw, y_test = gen_test()

    n_feats = X_train_raw.shape[1]
    #14, and 21 very correlated, can use one in exchange of the other if missing
    #TODO: make new features, combine 14 and 21
    #TODO: can missing data be predective?
    corr_pair = [14, 21]
    other_cols = [c for c in range(n_feats) if c not in corr_pair]

    hybrid_ct = ColumnTransformer(
        transformers=[
            ("iter", IterativeImputer(random_state=42, max_iter=10), corr_pair),
            ("med",  SimpleImputer(strategy="median"),                other_cols),
        ],
        verbose_feature_names_out=False
    )

    pipe = Pipeline([
        ("imputer", KNNImputer(n_neighbors=2)), 
        ("clf", HistGradientBoostingClassifier(
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            max_bins=255  
        ))
    ])
    median = SimpleImputer(strategy='median',)

    param_grid = [{
        "imputer": [median],
        "clf__max_depth": [7, 8, 9],
        "clf__learning_rate": [0.04, 0.05, 0.06],
        "clf__max_iter": [900, 1200],
        "clf__l2_regularization": [0.3, 0.5, 1.0],
        "clf__min_samples_leaf": [20, 30, 50],
    }]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring={"auc": "roc_auc", "tpr1": tpr1_scorer},
        refit="auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise"
    )

    grid.fit(X_train_raw, y_train)
    print("Best params:", grid.best_params_)
    print(f"Best CV TPR@1%: {grid.best_score_:.4f}")
    print(f"Best CV AUC (mean): {grid.cv_results_['mean_test_auc'][grid.best_index_]:.4f}")
    best_model = grid.best_estimator_
    p_test = best_model.predict_proba(X_test_raw)[:, 1]
    print(f"TEST AUC: {roc_auc_score(y_test, p_test):.4f}")
    print(f"TEST TPR@1%: {tpr_at_fpr_score(y_test, p_test, 0.01):.4f}")


def gen_train():
    train = pd.read_csv("spamTrain1.csv", header=None)
    X_train_raw, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    return X_train_raw.replace(-1, np.nan), y_train

def gen_test():
    test  = pd.read_csv("spamTrain2.csv",  header=None)
    X_test_raw,  y_test  = test.iloc[:,  :-1], test.iloc[:,  -1]
    return X_test_raw.replace(-1,  np.nan), y_test
if __name__ == '__main__':
    main()