#1
import pandas as pd
from sklearn.model_selection import train_test_split
from src.shipment_prep import FEATURES

table = pd.read_csv("data/modelling_table.csv")
X = table[FEATURES]                  # the 17 columns, pinned order
y = table["delivery_hours"]

# split FIRST, before any scaling or feature selection
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("train / validation / test:", len(X_train), len(X_val), len(X_test))
print("feature order ok:", list(X_train.columns) == FEATURES)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
#2
# ---------- Scaling: fit on the 840 training rows ONLY ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # learns mean and std from training rows
X_val_s = scaler.transform(X_val)           # transform only, never fit
X_test_s = scaler.transform(X_test)         # transform only, never fit

train_max = np.abs(X_train_s.mean(axis=0)).max()
test_max = np.abs(X_test_s.mean(axis=0)).max()

os.makedirs("outputs", exist_ok=True)
checks_b = pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows",
               "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [len(X_train), len(X_val), len(X_test), train_max, test_max],
}).round(4)
checks_b.to_csv("outputs/part_b_checks.csv", index=False)
print(checks_b.to_string(index=False))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Reusable metrics helper (all later models use this) ----------
def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }
#3
# ---------- Baseline: training mean for every validation row ----------
baseline_value = float(y_train.mean())              # learned from training rows only
baseline_pred = np.full(len(y_val), baseline_value)
baseline_metrics = regression_metrics(y_val, baseline_pred)

val_results = {"baseline_mean": baseline_metrics}   # later models get added here

print("baseline value (train mean):", round(baseline_value, 4))
print({k: round(v, 4) for k, v in baseline_metrics.items()})

#4
from sklearn.tree import DecisionTreeRegressor
depth_rows, best_depth, best_val = [], None, np.inf
for d in [1, 2, 3, 4, 6, 8, 12, None]:
    t = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    tr = np.sqrt(mean_squared_error(y_train, t.predict(X_train)))
    va = np.sqrt(mean_squared_error(y_val, t.predict(X_val)))
    depth_rows.append({"max_depth": "unlimited" if d is None else d, "train_rmse": tr, "val_rmse": va})
    if va < best_val - 1e-12:           # strict: ties keep the smaller depth
        best_val, best_depth = va, d
pd.DataFrame(depth_rows).round(4).to_csv("outputs/part_b_tree_depth.csv", index=False)

#5
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def metric_row(model_name, split, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"model": model_name, "split": split,
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred)}


candidates = {
    "linear_regression": Pipeline([("scaler", StandardScaler()),
                                   ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                   random_state=42),
}

metric_rows = [metric_row("mean_baseline", "val", y_val, baseline_pred)]
fitted = {}
for name, model in candidates.items():
    model.fit(X_train, y_train)                 # training rows only
    fitted[name] = model
    metric_rows.append(metric_row(name, "val", y_val, model.predict(X_val)))

metrics_df = pd.DataFrame(metric_rows)
metrics_df.round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print(metrics_df.round(4).to_string(index=False))

#6
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
    gb.fit(X_train, y_train)                     # training rows only
    pred = gb.predict(X_val)
    lr_rows.append({
        "learning_rate": lr,
        "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred))),
        "val_r2": r2_score(y_val, pred),
    })

lr_df = pd.DataFrame(lr_rows).round(4)
lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
print(lr_df.to_string(index=False))

#7
from sklearn.model_selection import KFold, cross_val_score

# ---------- Rank the four candidates by validation RMSE ----------
val_rows_df = metrics_df[(metrics_df["split"] == "val") & (metrics_df["model"] != "mean_baseline")]
ranking = val_rows_df.sort_values("RMSE")[["model", "RMSE", "R2"]].reset_index(drop=True)
print(ranking.round(4).to_string(index=False))
top_two = ranking["model"].head(2).tolist()
final_model_name = ranking["model"].iloc[0]        # lowest validation RMSE

# ---------- 5-fold CV on training + validation rows (1,020) ----------
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
assert len(X_trainval) == 1020

kf = KFold(n_splits=5, shuffle=True, random_state=42)   # ONE object, shared by both models
cv_rows = []
for name in top_two:
    scores = cross_val_score(candidates[name], X_trainval, y_trainval, cv=kf)   # default scoring = R2
    cv_rows.append({"model": name, "mean_r2": scores.mean(), "std_r2": scores.std()})

cv_df = pd.DataFrame(cv_rows).round(4)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)
print(cv_df.to_string(index=False))

# ---------- Final model: score ONCE on the test rows ----------
test_row = metric_row(final_model_name, "test", y_test, fitted[final_model_name].predict(X_test))
metrics_df = pd.concat([metrics_df, pd.DataFrame([test_row])], ignore_index=True)
metrics_df.round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print(metrics_df.round(4).to_string(index=False))
print("final model:", final_model_name)

assert (metrics_df["RMSE"] - np.sqrt(metrics_df["MSE"])).abs().max() < 0.001