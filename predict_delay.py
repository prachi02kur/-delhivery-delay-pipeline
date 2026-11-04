import os
import joblib
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features

# ---------- Load everything the script depends on ----------
new_raw = pd.read_csv("data/new_shipments.csv", encoding="utf-8")
hubs = pd.read_csv("data/hubs.csv", encoding="utf-8")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
impute = pd.read_csv("models/impute_values.csv")
medians = {row["column"]: float(row["median"]) for _, row in impute.iterrows()}
segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]

# ---------- Clean and build features with the stored medians ----------
clean, _ = clean_shipments(new_raw, medians)
feat = build_features(clean, hubs)
assert len(feat) == len(new_raw) == 48
assert feat[feature_order].isna().sum().sum() == 0

# ---------- Score ----------
X_new = feat[feature_order]                                    # columns in the saved training order
feat["delay_probability"] = model.predict_proba(X_new)[:, 1].round(4)
feat["flagged"] = (feat["delay_probability"] >= threshold).astype(int)

# ---------- Join the hub segment, decide the action ----------
feat = feat.merge(segments, on="hub_id", how="left", validate="many_to_one")
assert feat["segment"].notna().all()

feat["action"] = np.where(
    feat["flagged"] == 0, "No action",
    np.where(feat["segment"] == "Escalate", "Escalate to hub manager", "Proactive customer call"))

# ---------- Write, one row per new parcel in file order ----------
out = feat[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
os.makedirs("outputs", exist_ok=True)
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(out["action"].value_counts().to_string())
print("rows:", len(out), "| threshold used:", threshold)