import pandas as pd
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]                    # the 17 columns, pinned order
y = table["is_delayed"]                # classification target

# split FIRST; stratify on the target in BOTH calls
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("sizes train / val / test:", len(X_train), len(X_val), len(X_test))
print("delayed train / val / test:", int(y_train.sum()), int(y_val.sum()), int(y_test.sum()))
print("feature order ok:", list(X_train.columns) == FEATURES)

assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
assert (int(y_train.sum()), int(y_val.sum()), int(y_test.sum())) == (195, 42, 41)

import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def cm_row(model_name, y_true, y_pred):
    """One output row; every metric is computed from the four confusion-matrix cells."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0     # blind model: 0 by the brief
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"model": model_name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "accuracy": (tp + tn) / n, "precision": precision, "recall": recall, "f1": f1}


# ---------- Blind baseline: "not delayed" for everything ----------
rows = [cm_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int))]

# ---------- Logistic regression ----------
logit = Pipeline([("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=1000))])
logit.fit(X_train, y_train)
rows.append(cm_row("logistic_regression", y_val, logit.predict(X_val)))

# ---------- Decision tree: choose depth by validation F1 ----------
best_d, best_f1, best_tree = None, -1.0, None
for d in [2, 3, 4, 5, 6, 8, 10]:                       # ascending, so strict ">" keeps the smaller d on ties
    tree = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    f1 = round(cm_row("tmp", y_val, tree.predict(X_val))["f1"], 4)   # compare what the file will show
    if f1 > best_f1:
        best_d, best_f1, best_tree = d, f1, tree
rows.append(cm_row(f"decision_tree_depth_{best_d}", y_val, best_tree.predict(X_val)))

# ---------- Random forest, default settings ----------
forest_default = RandomForestClassifier(n_estimators=100, random_state=42)
forest_default.fit(X_train, y_train)
rows.append(cm_row("random_forest_default", y_val, forest_default.predict(X_val)))

models_df = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)
models_df.round(4).to_csv("outputs/part_c_models.csv", index=False)
print(models_df.round(4).to_string(index=False))
print("chosen tree depth:", best_d)

# self-check: cells must reproduce the metrics, and the cells must cover all 180 validation rows
assert (models_df[["TN", "FP", "FN", "TP"]].sum(axis=1) == len(y_val)).all()
assert ((models_df["TP"] + models_df["FN"]) == int(y_val.sum())).all()

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from src.shipment_prep import NOISE_COLUMNS

# ---------- Feature scores: training rows only ----------
rf_all = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)

scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": selector_all.scores_,
    "rf_importance": rf_all.feature_importances_,
})
scores_df["f_rank"] = scores_df["f_score"].rank(ascending=False, method="min").astype(int)
scores_df["is_noise"] = scores_df["feature"].isin(NOISE_COLUMNS)
scores_df = scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]]
scores_df.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
print(scores_df.sort_values("f_rank").round(4).to_string(index=False))

# ---------- Validation F1 for each k ----------
selectk_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe = Pipeline([("select", SelectKBest(f_classif, k=k)),
                     ("model", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe.fit(X_train, y_train)            # selector is fitted on training rows inside the pipeline
    selectk_rows.append({"k": k, "val_f1": f1_score(y_val, pipe.predict(X_val))})

selectk_df = pd.DataFrame(selectk_rows).round(4)
selectk_df.to_csv("outputs/part_c_selectk.csv", index=False)
print(selectk_df.to_string(index=False))

# ---------- From here on k = 7 (fixed by the brief) ----------
K_SELECTED = 7
selected_features = scores_df.sort_values("f_rank").head(K_SELECTED)["feature"].tolist()
print("k = 7 keeps:", selected_features)
print("noise columns among them:", [f for f in selected_features if f in NOISE_COLUMNS])

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# ---------- Selector for k = 7: fitted on the training rows only ----------
selector7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
X_train_sel = selector7.transform(X_train)
X_val_sel = selector7.transform(X_val)          # validation is transformed, never fitted on
print("columns kept:", [f for f, keep in zip(FEATURES, selector7.get_support()) if keep])

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

# ---------- Grid search: 27 candidates x 5 folds ----------
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_sel, y_train)

# ---------- Randomized search: 9 of the 27 candidates x 5 folds ----------
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,
                          n_iter=9, cv=5, random_state=42)
rand.fit(X_train_sel, y_train)

# ---------- Untuned forest on the same 7 columns ----------
forest_default7 = RandomForestClassifier(n_estimators=100, random_state=42)
forest_default7.fit(X_train_sel, y_train)

n_grid = len(grid.cv_results_["params"])        # counted from the search, not typed in
n_rand = len(rand.cv_results_["params"])

tuning_df = pd.DataFrame([
    {"search": "grid_search", "candidates": n_grid, "cv_folds": 5, "total_fits": n_grid * 5,
     "best_params": str(grid.best_params_), "best_cv_accuracy": grid.best_score_,
     "val_f1": f1_score(y_val, grid.best_estimator_.predict(X_val_sel))},
    {"search": "randomized_search", "candidates": n_rand, "cv_folds": 5, "total_fits": n_rand * 5,
     "best_params": str(rand.best_params_), "best_cv_accuracy": rand.best_score_,
     "val_f1": f1_score(y_val, rand.best_estimator_.predict(X_val_sel))},
    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "defaults", "best_cv_accuracy": np.nan,
     "val_f1": f1_score(y_val, forest_default7.predict(X_val_sel))},
])
tuning_df.round(4).to_csv("outputs/part_c_tuning.csv", index=False)
print(tuning_df.round(4).to_string(index=False))
print("grid CV accuracy >= randomized CV accuracy:", grid.best_score_ >= rand.best_score_)

#5
# ---------- Choose the final configuration from part_c_tuning.csv ----------
tuning_file = pd.read_csv("outputs/part_c_tuning.csv")

# tie-break order: default first, then grid, then randomized
priority = {"default_forest_same_7_features": 0, "grid_search": 1, "randomized_search": 2}
tuning_file["priority"] = tuning_file["search"].map(priority)
chosen = tuning_file.sort_values(["val_f1", "priority"], ascending=[False, True]).iloc[0]
chosen_name = chosen["search"]

# parameters kept in memory from the searches (no string parsing needed)
chosen_params = {
    "default_forest_same_7_features": {"n_estimators": 100},
    "grid_search": grid.best_params_,
    "randomized_search": rand.best_params_,
}[chosen_name]
print("chosen row:", chosen_name, "| params:", chosen_params, "| val_f1:", chosen["val_f1"])

# ---------- Final pipeline, fitted on the training rows ----------
final_clf = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
])
final_clf.fit(X_train, y_train)

# consistency check: should reproduce the val_f1 in the tuning file
check_f1 = round(f1_score(y_val, final_clf.predict(X_val)), 4)
print("pipeline val_f1:", check_f1, "| matches tuning file:", abs(check_f1 - chosen["val_f1"]) < 1e-4)

#6
# ---------- Alert threshold: validation probabilities of the final model ----------
val_proba = final_clf.predict_proba(X_val)[:, 1]       # positive class = delayed
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # literal list, exact decimals

thr_rows = []
for t in thresholds:
    pred = (val_proba >= t).astype(int)                  # delayed when probability >= t
    row = cm_row("tmp", y_val, pred)
    row.pop("model")
    thr_rows.append({"threshold": t, **row})

thr_df = pd.DataFrame(thr_rows)
thr_df.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
print(thr_df.round(4).to_string(index=False))

# operations rule: highest t whose validation recall is at least 0.80; if none, use 0.1
ok_thr = thr_df.loc[thr_df["recall"] >= 0.80, "threshold"]
ALERT_THRESHOLD = float(ok_thr.max()) if len(ok_thr) > 0 else 0.1
print("alert threshold:", ALERT_THRESHOLD)
print(thr_df.loc[thr_df["threshold"] == ALERT_THRESHOLD, ["recall", "precision", "FP", "FN"]].round(4).to_string(index=False))

#7
# ---------- ONE test evaluation at the chosen threshold ----------
test_proba = final_clf.predict_proba(X_test)[:, 1]            # the only place X_test is scored
test_pred = (test_proba >= ALERT_THRESHOLD).astype(int)

test_row = cm_row("tmp", y_test, test_pred)
test_row.pop("model")
final_test_df = pd.DataFrame([{"threshold": ALERT_THRESHOLD, **test_row}])
final_test_df.round(4).to_csv("outputs/part_c_final_test.csv", index=False)
print(final_test_df.round(4).to_string(index=False))

# ---------- Final configuration, read from the fitted pipeline ----------
rf_params = final_clf.named_steps["model"].get_params()
max_depth = rf_params["max_depth"]
config_df = pd.DataFrame([{
    "selector_k": final_clf.named_steps["selector"].k,
    "n_estimators": rf_params["n_estimators"],
    "max_depth": "unlimited" if max_depth is None else max_depth,
    "min_samples_leaf": rf_params["min_samples_leaf"],
    "threshold": ALERT_THRESHOLD,
    "chosen_from": chosen_name,               # name of the winning row in part_c_tuning.csv
}])
config_df.to_csv("outputs/part_c_final_config.csv", index=False)
print(config_df.to_string(index=False))

assert test_row["TP"] + test_row["FN"] == int(y_test.sum())