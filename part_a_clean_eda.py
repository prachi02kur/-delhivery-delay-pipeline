import os
import numpy as np
import pandas as pd
from src.shipment_prep import clean_shipments, build_features, FEATURES, PROMISED_HOURS

os.makedirs("outputs", exist_ok=True)
raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean, medians = clean_shipments(raw)
table = build_features(clean, hubs)
table["is_delayed"] = (table["delivery_hours"] > table["promised_hours"]).astype(int)

table.to_csv("data/modelling_table.csv", index=False)

dedup = raw.drop_duplicates()
declared_num = pd.to_numeric(
    dedup["declared_value"].astype("string").str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
    errors="coerce")
weight_before = dedup["parcel_weight_kg"]
spell_before = dedup["service_type"].nunique()

needed = FEATURES + ["delivery_hours", "is_delayed"]

q1 = table["delivery_hours"].quantile(0.25)
q3 = table["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

checks = {
    "rows_raw": len(raw),
    "duplicate_rows_removed": len(raw) - len(dedup),
    "rows_after_dedup": len(dedup),
    "missing_distance_km_before": dedup["distance_km"].isna().sum(),
    "missing_parcel_weight_before": dedup["parcel_weight_kg"].isna().sum(),   # excludes sentinel rows
    "missing_declared_value_before": dedup["declared_value"].isna().sum(),
    "missing_payment_mode_before": dedup["payment_mode"].isna().sum(),
    "weight_sentinel_rows": (dedup["parcel_weight_kg"] > 100).sum(),
    "service_type_spellings_before": dedup["service_type"].nunique(),
    "service_type_spellings_after": dedup["service_type"].str.strip().str.title().nunique(),
    "nulls_in_features_and_target": table[needed].isna().sum().sum(),
    "declared_value_mean_inr": table["declared_value_inr"].mean(),
    "declared_value_median_inr": table["declared_value_inr"].median(),
    "log_declared_value_mean": table["log_declared_value"].mean(),
    "log_declared_value_median": table["log_declared_value"].median(),
    "hours_q1": q1,
    "hours_q3": q3,
    "hours_iqr": iqr,
    "hours_lower_bound": lower,
    "hours_upper_bound": upper,
    "hours_outliers_below": (table["delivery_hours"] < lower).sum(),
    "hours_outliers_above": (table["delivery_hours"] > upper).sum(),
}

checks_df = pd.DataFrame({"metric": list(checks.keys()),
                          "value": [float(v) for v in checks.values()]}).round(4)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)
print(checks_df.to_string(index=False))

# Delay evidence by group
grp_src = table.assign(
    distance_band=pd.cut(table["distance_km"],
                         bins=[0, 200, 400, 600, 2000],
                         labels=["0-200", "201-400", "401-600", "601+"],
                         right=True)
)


def group_summary(df, col, order):
    g = df.groupby(col, observed=False).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        mean_hours=("delivery_hours", "mean"),
    )
    g = g.reindex(order)
    g["delay_rate"] = g["delayed"] / g["shipments"]
    g = g.reset_index().rename(columns={col: "group"})
    g.insert(0, "dimension", col)
    g["group"] = g["group"].astype(str)
    return g[["dimension", "group", "shipments", "delayed", "delay_rate", "mean_hours"]]


by_group = pd.concat([
    group_summary(grp_src, "service_type", ["Express", "Standard", "Economy"]),
    group_summary(grp_src, "hub_tier", [1, 2, 3]),
    group_summary(grp_src, "distance_band", ["0-200", "201-400", "401-600", "601+"]),
], ignore_index=True).round(4)

by_group.to_csv("outputs/part_a_delay_by_group.csv", index=False)
print(by_group.to_string(index=False))

# self-check: every dimension must cover all 1,200 shipments and the same delayed total
print(by_group.groupby("dimension")[["shipments", "delayed"]].sum())

# ---------- Correlation matrix ----------
corr = table.select_dtypes(include="number").corr().round(4)
corr.to_csv("outputs/part_a_correlations.csv", index=True)   # row labels kept as first column

print("correlation matrix shape:", corr.shape)

# for README finding (iv): top 3 by absolute correlation with delivery_hours
target_corr = corr["delivery_hours"].drop("delivery_hours")
top3 = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index).head(3)
print(top3.map(lambda v: f"{v:.4f}"))