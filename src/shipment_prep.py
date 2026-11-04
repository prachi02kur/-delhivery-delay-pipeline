import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no",
                 "sticker_serial_mod", "shelf_slot_no"]

MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]

def clean_shipments(raw, medians=None):
    """Clean a raw shipments export. Returns (clean_df, medians)."""
    df = raw.copy()

    # 1. drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. service_type: strip spaces, title case
    df["service_type"] = df["service_type"].str.strip().str.title()

    # 3. payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value text -> declared_value_inr number
    cleaned = (df["declared_value"].str.replace("₹", "", regex=False)
                                   .str.replace(",", "", regex=False))
    df["declared_value_inr"] = pd.to_numeric(cleaned, errors="coerce")
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time -> datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. weight sentinel -> missing
    df.loc[df["parcel_weight_kg"] > 100 , "parcel_weight_kg"] = np.nan

    # 7. median fill (compute only if not supplied)
    if medians is None:
        medians = {c: df[c].median() for c in MEDIAN_COLUMNS}
    medians = {c: float(medians[c]) for c in MEDIAN_COLUMNS}
    for c in MEDIAN_COLUMNS:
        df[c] = df[c].fillna(medians[c])

    return df, medians

def build_features(clean, hubs):
    """Attach hub data and build all required feature columns without using delivery_hours."""
    df = clean.copy()
    
    # 1. Merge hub metadata on hub_id
    df = df.merge(hubs, on="hub_id", how="left")
    
    # 2. promised_hours from PROMISED_HOURS mapping
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    
    # 3. is_cod: 1 when payment_mode is COD, else 0 (Unknown gives 0)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    
    # 4. is_weekend: 1 when dispatch day is Saturday or Sunday
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    
    # 5. dispatch_hour: integer hour of the timestamp
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    
    # 6. log_declared_value: natural log of declared_value_inr
    df["log_declared_value"] = np.log1p(df["declared_value_inr"])
    
    return df