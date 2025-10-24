# classifySpam.py

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # <-- this line is required
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
def predictTest(trainFeatures, trainLabels, testFeatures):
    """
    Predict probabilities on testFeatures using a stacked model:
    Base learners: XGBoost + Logistic Regression
    Final estimator: Logistic Regression
    Fixed XGBoost parameters as specified.
    """

    # ---- Impute missing values ----
    trainFeatures[trainFeatures==-1]=np.nan
    testFeatures[testFeatures==-1]=np.nan
    #imputer = IterativeImputer(max_iter=10, random_state=random_state)
    X_train_imputed = trainFeatures
    X_test_imputed = testFeatures

    # ---- Standard scaling ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # ---- Fixed XGBoost parameters ----
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1
    )

    # ---- Logistic Regression base learner ----
    #lr_base = LogisticRegression(solver="liblinear", random_state=random_state)

    # ---- Stacking classifier ----
    stack = xgb
    # ---- Fit stacking model ----
    stack.fit(X_train_scaled, trainLabels)

    # ---- Predict probabilities for test set ----
    test_proba = stack.predict_proba(X_test_scaled)[:, 1]

    return test_proba
