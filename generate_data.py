"""
generate_data.py
----------------
Generates synthetic hospital supply chain procurement data across 8 facilities.
Produces realistic order, inventory, delivery, and stockout records that mimic
real-world patterns including seasonality, supplier variability, and facility size effects.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json

SEED = 42
rng = np.random.default_rng(SEED)

# ── Facility definitions ──────────────────────────────────────────────────────
FACILITIES = {
    "F001": {"name": "City General Hospital",     "beds": 450, "region": "Northeast"},
    "F002": {"name": "Riverside Medical Center",  "beds": 280, "region": "Southeast"},
    "F003": {"name": "Valley Community Hospital", "beds": 180, "region": "Midwest"},
    "F004": {"name": "Lakeside Health System",    "beds": 620, "region": "Midwest"},
    "F005": {"name": "Summit Regional Medical",   "beds": 310, "region": "West"},
    "F006": {"name": "Coastal Care Hospital",     "beds": 390, "region": "West"},
    "F007": {"name": "Northgate Medical Center",  "beds": 520, "region": "Northeast"},
    "F008": {"name": "Pinebrook Community Hosp.", "beds": 140, "region": "Southeast"},
}

# ── Supplier definitions ──────────────────────────────────────────────────────
SUPPLIERS = {
    "S01": {"name": "MedSupply Pro",       "reliability": 0.97, "avg_lead_days": 3.0},
    "S02": {"name": "HealthCore Logistics", "reliability": 0.91, "avg_lead_days": 5.0},
    "S03": {"name": "CareChain Inc.",       "reliability": 0.85, "avg_lead_days": 4.5},
    "S04": {"name": "Nexus Medical Supply", "reliability": 0.78, "avg_lead_days": 7.0},
    "S05": {"name": "Allied Health Dist.",  "reliability": 0.93, "avg_lead_days": 3.5},
}

# ── Product definitions ───────────────────────────────────────────────────────
PRODUCTS = {
    "P001": {"name": "Surgical Gloves (L, 100ct)",    "category": "PPE",          "unit_cost": 12.50,  "critical": True},
    "P002": {"name": "N95 Respirators (20ct)",        "category": "PPE",          "unit_cost": 38.00,  "critical": True},
    "P003": {"name": "IV Saline Solution 0.9% 1L",   "category": "Fluids",       "unit_cost": 4.20,   "critical": True},
    "P004": {"name": "Syringes 10mL (100ct)",         "category": "Disposables",  "unit_cost": 9.80,   "critical": False},
    "P005": {"name": "Surgical Masks (50ct)",         "category": "PPE",          "unit_cost": 15.00,  "critical": False},
    "P006": {"name": "Blood Collection Tubes (100ct)","category": "Lab",          "unit_cost": 22.00,  "critical": False},
    "P007": {"name": "Nitrile Exam Gloves (M, 100ct)","category": "PPE",          "unit_cost": 11.00,  "critical": True},
    "P008": {"name": "Gauze Pads 4x4 (200ct)",       "category": "Wound Care",   "unit_cost": 7.50,   "critical": False},
    "P009": {"name": "Catheter Kit 16Fr",             "category": "Procedures",   "unit_cost": 18.50,  "critical": True},
    "P010": {"name": "Alcohol Swabs (200ct)",         "category": "Disposables",  "unit_cost": 5.80,   "critical": False},
    "P011": {"name": "Insulin Syringes 1mL (100ct)", "category": "Disposables",  "unit_cost": 14.20,  "critical": True},
    "P012": {"name": "Oxygen Mask Standard",          "category": "Respiratory",  "unit_cost": 8.90,   "critical": True},
}

SUPPLIER_PRODUCT_MAP = {
    "P001": ["S01", "S02", "S03"],
    "P002": ["S01", "S04", "S05"],
    "P003": ["S02", "S03", "S05"],
    "P004": ["S01", "S02", "S04"],
    "P005": ["S01", "S03", "S05"],
    "P006": ["S02", "S04"],
    "P007": ["S01", "S02", "S05"],
    "P008": ["S02", "S03", "S04"],
    "P009": ["S01", "S05"],
    "P010": ["S01", "S02", "S03", "S04"],
    "P011": ["S01", "S05"],
    "P012": ["S03", "S04", "S05"],
}


def seasonal_demand_multiplier(date: datetime) -> float:
    """Winter surge (flu season) + mild summer bump."""
    doy = date.timetuple().tm_yday
    base = 1.0 + 0.25 * np.cos(2 * np.pi * (doy - 355) / 365)   # peak Jan 1
    return float(base)


def generate_orders(start: datetime, end: datetime) -> pd.DataFrame:
    """Generate one order record per (facility, product, week)."""
    records = []
    current = start
    order_id = 1

    while current <= end:
        for fac_id, fac in FACILITIES.items():
            scale = fac["beds"] / 300          # normalize to ~300-bed baseline
            season = seasonal_demand_multiplier(current)

            for prod_id, prod in PRODUCTS.items():
                # Base weekly demand proportional to facility size + noise
                base_demand = scale * season * rng.integers(10, 60)
                quantity_ordered = max(1, int(base_demand))

                suppliers = SUPPLIER_PRODUCT_MAP[prod_id]
                supplier_id = rng.choice(suppliers)
                supplier = SUPPLIERS[supplier_id]

                # Lead time with supplier-specific variability
                lead_days = max(
                    1,
                    int(rng.normal(supplier["avg_lead_days"], supplier["avg_lead_days"] * 0.3))
                )

                # Delivery success / partial delivery
                delivered = rng.random() < supplier["reliability"]
                if delivered:
                    quantity_received = quantity_ordered
                    delay_days = 0
                else:
                    # Partial or delayed
                    partial = rng.random() < 0.5
                    if partial:
                        quantity_received = int(quantity_ordered * rng.uniform(0.4, 0.85))
                        delay_days = rng.integers(2, 8)
                    else:
                        quantity_received = 0
                        delay_days = rng.integers(5, 21)

                order_date = current
                expected_delivery = order_date + timedelta(days=lead_days)
                actual_delivery = expected_delivery + timedelta(days=int(delay_days))

                records.append({
                    "order_id":          f"ORD-{order_id:06d}",
                    "facility_id":       fac_id,
                    "facility_name":     fac["name"],
                    "region":            fac["region"],
                    "facility_beds":     fac["beds"],
                    "product_id":        prod_id,
                    "product_name":      prod["name"],
                    "category":          prod["category"],
                    "critical_item":     prod["critical"],
                    "unit_cost":         prod["unit_cost"],
                    "supplier_id":       supplier_id,
                    "supplier_name":     SUPPLIERS[supplier_id]["name"],
                    "supplier_reliability": supplier["reliability"],
                    "order_date":        order_date.strftime("%Y-%m-%d"),
                    "expected_delivery": expected_delivery.strftime("%Y-%m-%d"),
                    "actual_delivery":   actual_delivery.strftime("%Y-%m-%d"),
                    "quantity_ordered":  quantity_ordered,
                    "quantity_received": quantity_received,
                    "lead_days_planned": lead_days,
                    "delay_days":        int(delay_days),
                    "fulfilled":         delivered,
                    "unit_cost_total":   round(quantity_ordered * prod["unit_cost"], 2),
                })
                order_id += 1

        current += timedelta(weeks=1)

    return pd.DataFrame(records)


def generate_inventory_snapshots(orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate weekly inventory levels per facility/product using a simple
    consumption model: consume ~80% of received stock each week.
    """
    records = []
    dates = sorted(orders_df["order_date"].unique())

    # Starting inventory: random 2–6 weeks of stock per facility/product
    inventory = {}
    for fac_id in FACILITIES:
        for prod_id in PRODUCTS:
            inventory[(fac_id, prod_id)] = rng.integers(20, 120)

    for date in dates:
        week_orders = orders_df[orders_df["order_date"] == date]

        for fac_id in FACILITIES:
            for prod_id, prod in PRODUCTS.items():
                prev = inventory.get((fac_id, prod_id), 0)

                # Receive stock from fulfilled orders this week
                received = week_orders[
                    (week_orders["facility_id"] == fac_id) &
                    (week_orders["product_id"] == prod_id)
                ]["quantity_received"].sum()

                # Weekly consumption
                scale = FACILITIES[fac_id]["beds"] / 300
                consumption = int(rng.normal(scale * 30, scale * 8))
                consumption = max(0, consumption)

                closing = max(0, prev + int(received) - consumption)
                stockout = int(closing == 0)

                # Safety stock threshold (2 weeks of average consumption)
                safety_stock = int(scale * 60)
                below_safety = int(closing < safety_stock)

                records.append({
                    "snapshot_date":   date,
                    "facility_id":     fac_id,
                    "product_id":      prod_id,
                    "opening_stock":   prev,
                    "received":        int(received),
                    "consumption":     consumption,
                    "closing_stock":   closing,
                    "safety_stock":    safety_stock,
                    "stockout":        stockout,
                    "below_safety":    below_safety,
                    "critical_item":   prod["critical"],
                })

                inventory[(fac_id, prod_id)] = closing

    return pd.DataFrame(records)


def build_ml_dataset(orders_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a feature-rich dataset for stockout prediction.
    Label: will this facility/product experience a stockout within the next 14 days?
    """
    inv = inventory_df.copy()
    inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
    inv = inv.sort_values(["facility_id", "product_id", "snapshot_date"])

    # Rolling features (4-week window)
    grp = inv.groupby(["facility_id", "product_id"])
    inv["avg_consumption_4w"]  = grp["consumption"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    inv["avg_received_4w"]     = grp["received"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    inv["stockout_rate_4w"]    = grp["stockout"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    inv["below_safety_rate_4w"]= grp["below_safety"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    inv["days_of_supply"]      = (inv["closing_stock"] / inv["avg_consumption_4w"].replace(0, 1)).clip(upper=60)

    # Merge supplier reliability from orders
    supplier_stats = (
        orders_df.groupby(["facility_id", "product_id", "order_date"])
        .agg(
            avg_supplier_reliability=("supplier_reliability", "mean"),
            avg_delay_days=("delay_days", "mean"),
            fulfillment_rate=("fulfilled", "mean"),
        )
        .reset_index()
        .rename(columns={"order_date": "snapshot_date"})
    )
    supplier_stats["snapshot_date"] = pd.to_datetime(supplier_stats["snapshot_date"])

    ml = inv.merge(supplier_stats, on=["facility_id", "product_id", "snapshot_date"], how="left")

    # Add facility beds
    bed_map = {k: v["beds"] for k, v in FACILITIES.items()}
    ml["facility_beds"] = ml["facility_id"].map(bed_map)

    # Forward-looking label: stockout in next 2 weeks
    ml = ml.sort_values(["facility_id", "product_id", "snapshot_date"])
    ml["stockout_next_14d"] = (
        grp["stockout"]
        .transform(lambda x: x.shift(-1).fillna(0).astype(int) | x.shift(-2).fillna(0).astype(int))
    )

    # Season feature
    ml["week_of_year"] = ml["snapshot_date"].dt.isocalendar().week.astype(int)

    # Drop rows with NaN labels
    ml = ml.dropna(subset=["stockout_next_14d"])
    ml["stockout_next_14d"] = ml["stockout_next_14d"].astype(int)

    return ml


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    START = datetime(2023, 1, 2)
    END   = datetime(2024, 12, 30)

    print("Generating order records...")
    orders = generate_orders(START, END)
    orders.to_csv("data/orders.csv", index=False)
    print(f"  → {len(orders):,} order records saved to data/orders.csv")

    print("Simulating inventory snapshots...")
    inventory = generate_inventory_snapshots(orders)
    inventory.to_csv("data/inventory_snapshots.csv", index=False)
    print(f"  → {len(inventory):,} inventory records saved to data/inventory_snapshots.csv")

    print("Building ML dataset...")
    ml_data = build_ml_dataset(orders, inventory)
    ml_data.to_csv("data/ml_dataset.csv", index=False)
    print(f"  → {len(ml_data):,} ML samples saved to data/ml_dataset.csv")

    # Save metadata
    meta = {
        "facilities": FACILITIES,
        "suppliers":  SUPPLIERS,
        "products":   PRODUCTS,
        "date_range": {"start": START.strftime("%Y-%m-%d"), "end": END.strftime("%Y-%m-%d")},
    }
    with open("data/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("  → Metadata saved to data/metadata.json")
    print("\nData generation complete.")
