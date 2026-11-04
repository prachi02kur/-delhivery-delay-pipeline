#1
import os
import pandas as pd
import numpy as np

table = pd.read_csv("data/modelling_table.csv")

profile = (table.groupby("hub_id")
           .agg(shipments=("shipment_id", "count"),
                mean_delivery_hours=("delivery_hours", "mean"),
                delay_rate=("is_delayed", "mean"),        # share of delayed parcels
                mean_distance_km=("distance_km", "mean"),
                mean_n_stops=("n_stops", "mean"))
           .reset_index()
           .sort_values("hub_id")
           .reset_index(drop=True))

os.makedirs("outputs", exist_ok=True)
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)

print(profile.round(4).to_string(index=False))
print("rows:", len(profile), "| shipments total:", int(profile["shipments"].sum()))
assert len(profile) == 24 and profile["shipments"].sum() == len(table)
#2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

CLUSTER_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]

# profile is already sorted by ascending hub_id (HUB01 first, HUB24 last)
assert profile["hub_id"].tolist() == sorted(profile["hub_id"].tolist())

scaler_hubs = StandardScaler()
profile_scaled = scaler_hubs.fit_transform(profile[CLUSTER_COLUMNS])   # 24 hubs, fixed order

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
profile["cluster"] = kmeans.fit_predict(profile_scaled)

print(profile[["hub_id", "cluster"] + CLUSTER_COLUMNS].round(4).to_string(index=False))
print("hubs per cluster:", profile["cluster"].value_counts().sort_index().to_dict())
print("inertia:", round(float(kmeans.inertia_), 4))

#3
from itertools import combinations
from scipy.spatial.distance import pdist

# all 276 unordered hub pairs, on the scaled matrix (hub order HUB01..HUB24)
pair_dist = pdist(profile_scaled, metric="euclidean")
pair_idx = list(combinations(range(len(profile_scaled)), 2))    # same order that pdist uses
assert len(pair_idx) == 276 == len(pair_dist)

sep_rows = []
for K in [2, 3, 4, 5]:
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(profile_scaled)

    same = np.array([labels[i] == labels[j] for i, j in pair_idx])
    across_mean = pair_dist[~same].mean()
    within_mean = pair_dist[same].mean()

    sizes = sorted(np.bincount(labels).tolist(), reverse=True)
    sep_rows.append({"K": K,
                     "across_over_within": across_mean / within_mean,
                     "cluster_sizes": "-".join(str(s) for s in sizes)})

sep_df = pd.DataFrame(sep_rows)
sep_df.round(4).to_csv("outputs/part_d_separation.csv", index=False)
print(sep_df.round(4).to_string(index=False))
#4
# ---------- Name the clusters by mean hub delay_rate ----------
cluster_delay = profile.groupby("cluster")["delay_rate"].mean().sort_values()   # lowest first
segment_names = ["Standard", "Watch", "Escalate"]
cluster_to_segment = dict(zip(cluster_delay.index, segment_names))
profile["segment"] = profile["cluster"].map(cluster_to_segment)
print("cluster -> segment:", cluster_to_segment)

# ---------- hub_segments.csv ----------
hubs = pd.read_csv("data/hubs.csv", encoding="utf-8")
hub_segments = (profile[["hub_id", "segment"]]
                .merge(hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left", validate="one_to_one")
                .sort_values("hub_id").reset_index(drop=True))
assert len(hub_segments) == 24 and hub_segments["hub_city"].notna().all()
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

# ---------- segment_profile.csv: means of the hub-level values ----------
segment_profile = (profile.groupby("segment")
                   .agg(hubs=("hub_id", "count"),
                        mean_delivery_hours=("mean_delivery_hours", "mean"),
                        delay_rate=("delay_rate", "mean"),
                        mean_distance_km=("mean_distance_km", "mean"),
                        mean_n_stops=("mean_n_stops", "mean"))
                   .reindex(segment_names).reset_index())
segment_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)

# ---------- tier_vs_segment_mismatch.csv ----------
tier_implies = {1: "Standard", 2: "Watch", 3: "Escalate"}
mismatch = hub_segments[hub_segments["segment"] != hub_segments["hub_tier"].map(tier_implies)]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

print(hub_segments.to_string(index=False))
print(segment_profile.round(4).to_string(index=False))
print("mismatches:", len(mismatch))
print(mismatch.to_string(index=False))