# Devak Pardasani - Full Robust Stacking + GridSearchCV + Base Learner Plot

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def gen_combined_data(test_size=0.2, random_state=42):
    # Load both datasets
    df1 = pd.read_csv("spamTrain1.csv", header=None)
    df2 = pd.read_csv("spamTrain2.csv", header=None)

    # Concatenate
    df = pd.concat([df1, df2], ignore_index=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
def tpr_at_fpr_score(y_true, y_proba, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_proba, drop_intermediate=False)
    return float(np.interp(target_fpr, fpr, tpr))


def gen_train():
    train = pd.read_csv("spamTrain1.csv", header=None)
    return train.iloc[:, :-1], train.iloc[:, -1]


def gen_test():
    test = pd.read_csv("spamTrain2.csv", header=None)
    return test.iloc[:, :-1], test.iloc[:, -1]



def tpr_at_fpr_score(y_true, y_proba, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(y_true, y_proba, drop_intermediate=False)
    return float(np.interp(target_fpr, fpr, tpr))

# ... (imports and tpr_at_fpr_score unchanged)


def run_experiment(random_state=42):
    print(f"\n===== Running for random_state={random_state} =====")
    start_time = time.time()

    # ---- Load combined data (80/20 split) ----
    X_train_raw, y_train = gen_train()
    X_test_raw , y_test = gen_test();

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ---- Impute ----
    imputer = IterativeImputer(max_iter=10, random_state=random_state)
    X_train_imputed = imputer.fit_transform(X_train_raw)
    X_test_imputed = imputer.transform(X_test_raw)

    # ---- Standard scaling ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    # ---- GridSearchCV for XGB ----
    pipe_xgb = Pipeline([
        ("clf", XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
            scale_pos_weight=1.415,
            objective="binary:logistic",
            n_jobs=-1
        ))
    ])

    param_grid = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__min_child_weight": [1, 3],
        "clf__gamma": [0, 0.1]
    }

    grid_search = GridSearchCV(
        estimator=pipe_xgb,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)
    best_xgb = grid_search.best_estimator_.named_steps["clf"]

    print(f"Best XGB Params ({random_state}): {grid_search.best_params_}")
    print(f"Best CV AUC ({random_state}): {grid_search.best_score_:.4f}")

    # ---- Stacking (XGB + LR) → final LR ----
    stack = StackingClassifier(
        estimators=[
            ("xgb", best_xgb),
            ("lr", LogisticRegression(solver="liblinear", random_state=random_state))
        ],
        final_estimator=LogisticRegression(solver="liblinear", random_state=random_state),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1
    )

    stack.fit(X_train_scaled, y_train)

    # ---- Create meta-features ----
    xgb_train_preds = stack.named_estimators_["xgb"].predict_proba(X_train_scaled)[:, 1]
    lr_train_preds = stack.named_estimators_["lr"].predict_proba(X_train_scaled)[:, 1]
    X_meta_train = np.column_stack((xgb_train_preds, lr_train_preds))
    y_meta_train = y_train

    xgb_test_preds = stack.named_estimators_["xgb"].predict_proba(X_test_scaled)[:, 1]
    lr_test_preds = stack.named_estimators_["lr"].predict_proba(X_test_scaled)[:, 1]
    X_meta_test = np.column_stack((xgb_test_preds, lr_test_preds))
    y_meta_test = y_test

    # ---- Train final Logistic Regression ----
    final_lr = stack.final_estimator_
    final_lr.fit(X_meta_train, y_meta_train)

    # ---- Decision boundary for plotting ----
    xx, yy = np.meshgrid(np.linspace(0, 1, 300), np.linspace(0, 1, 300))
    Z = final_lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # ---- Plot train meta-features ----
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_meta_train[:,0], X_meta_train[:,1], c=y_meta_train, cmap="coolwarm", alpha=0.7)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black')
    if len(np.unique(y_meta_train)) > 1:
        plt.colorbar(scatter, label="Target")
    plt.xlabel("XGB predictions")
    plt.ylabel("LR predictions")
    plt.title("Training: Base Learner Predictions with LR Boundary")
    plt.show()

    # ---- Plot test meta-features ----
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_meta_test[:,0], X_meta_test[:,1], c=y_meta_test, cmap="coolwarm", alpha=0.7)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black')
    if len(np.unique(y_meta_test)) > 1:
        plt.colorbar(scatter, label="Target")
    plt.xlabel("XGB predictions")
    plt.ylabel("LR predictions")
    plt.title("Test: Base Learner Predictions with LR Boundary")
    plt.show()

    # ---- Evaluate ----
    p_test = stack.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    tpr = tpr_at_fpr_score(y_test, p_test, 0.01)
    runtime = time.time() - start_time

    print(f"TEST AUC ({random_state}): {auc:.4f}")
    print(f"TEST TPR@1% ({random_state}): {tpr:.4f}")
    print(f"Runtime: {runtime/60:.2f} minutes")

    return auc, tpr, grid_search.best_params_, runtime



def test_random_state_robustness(seeds=[42, 100, 2024, 7, 99]):
    results, params = [], []

    for seed in seeds:
        auc, tpr, best_params, runtime = run_experiment(random_state=seed)
        results.append({"Seed": seed, "AUC": auc, "TPR@1%": tpr, "Runtime_min": runtime/60})
        # Save best parameters per seed
        params.append({"Seed": seed, **best_params})

    df_results = pd.DataFrame(results)
    df_params = pd.DataFrame(params)

    print("\n=== Robustness Summary ===")
    print(df_results)
    print(f"\nAUC mean ± std : {df_results['AUC'].mean():.4f} ± {df_results['AUC'].std():.4f}")
    print(f"TPR@1% mean ± std : {df_results['TPR@1%'].mean():.4f} ± {df_results['TPR@1%'].std():.4f}")

    sns.boxplot(data=df_results[["AUC", "TPR@1%"]])
    plt.title("Model Robustness to random_state (Train=spamTrain1, Test=spamTrain2)")
    plt.show()

    # ---- Save to CSV ----
    df_results.to_csv("robustness_results.csv", index=False)
    df_params.to_csv("best_params_per_seed.csv", index=False)
    print("\nResults saved to 'robustness_results.csv' and 'best_params_per_seed.csv'.")

    return df_results, df_params

if __name__ == "__main__":
    df_results, df_params = test_random_state_robustness(
        seeds=[42, 100, 2024, 7, 99]
    )
