import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------- Read the final configuration written by Part C ----------
cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(cfg["max_depth"])
threshold = float(cfg["threshold"])

# ---------- Rebuild Part C's stratified split with the same settings ----------
table = pd.read_csv("data/modelling_table.csv")
X, y = table[FEATURES], table["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)

# ---------- Fit the final pipeline on the training rows ----------
final_pipe = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42,
                                     n_estimators=int(cfg["n_estimators"]),
                                     max_depth=max_depth,
                                     min_samples_leaf=int(cfg["min_samples_leaf"]))),
])
final_pipe.fit(X_train, y_train)

# ---------- Save the four artefacts ----------
joblib.dump(final_pipe, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv", encoding="utf-8")
_, medians = clean_shipments(raw)                      # the three fill-in medians from the full table
pd.DataFrame({"column": list(medians.keys()),
              "median": list(medians.values())}).to_csv("models/impute_values.csv", index=False)

# ---------- Reload check on the 180 test rows (not an evaluation: no metric is computed) ----------
reloaded = joblib.load("models/delay_model.pkl")
p_mem = final_pipe.predict_proba(X_test)[:, 1]
p_disk = reloaded.predict_proba(X_test)[:, 1]

check = pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
    "value": [len(p_disk), float(np.abs(p_mem - p_disk).max()), threshold],
}).round(4)
check.to_csv("outputs/part_d_reload_check.csv", index=False)
print(check.to_string(index=False))
print("saved:", sorted(os.listdir("models")))