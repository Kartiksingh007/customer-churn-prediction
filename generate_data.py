import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

N_CUSTOMERS = 2500
MAX_ORDERS = 18
REFERENCE_DATE = pd.Timestamp("2026-09-30")

customers = []
rows = []

segments = ["Low", "Medium", "High"]

for customer_id in range(10001, 10001 + N_CUSTOMERS):
    # Latent customer behaviour controls both transactions and churn.
    segment = rng.choice(segments, p=[0.35, 0.45, 0.20])

    if segment == "High":
        order_count = int(rng.integers(8, MAX_ORDERS + 1))
        spend_base = rng.uniform(800, 1800)
        activity_days = rng.integers(10, 45)
    elif segment == "Medium":
        order_count = int(rng.integers(4, 12))
        spend_base = rng.uniform(400, 1000)
        activity_days = rng.integers(20, 100)
    else:
        order_count = int(rng.integers(1, 8))
        spend_base = rng.uniform(150, 600)
        activity_days = rng.integers(45, 180)

    # Some customers are intentionally inactive, making churn learnable.
    risk = (
        0.18
        + (segment == "Low") * 0.15
        + (activity_days > 90) * 0.25
        + rng.normal(0, 0.05)
    )
    risky = rng.random() < min(max(risk, 0.02), 0.90)

    last_gap = (
        rng.integers(90, 180)
        if risky
        else rng.integers(2, min(activity_days + 1, 60))
    )

    last_date = REFERENCE_DATE - pd.Timedelta(days=int(last_gap))

    # Earlier purchases distributed before the last purchase.
    for order_no in range(order_count):
        if order_no == order_count - 1:
            order_date = last_date
        else:
            days_before_last = int(rng.integers(1, max(2, activity_days + 1)))
            order_date = last_date - pd.Timedelta(days=days_before_last)

        quantity = int(rng.integers(1, 6))
        unit_price = round(max(50, rng.normal(spend_base / 2, spend_base / 8)), 2)
        total = round(quantity * unit_price, 2)

        rows.append(
            {
                "customer_id": customer_id,
                "order_id": f"ORD{customer_id}_{order_no+1:03d}",
                "order_date": order_date.strftime("%Y-%m-%d"),
                "quantity": quantity,
                "unit_price": unit_price,
                "total_revenue": total,
                "category": rng.choice(["Electronics", "Fashion", "Home", "Beauty", "Grocery"]),
                "sales_channel": rng.choice(["Online", "Store"], p=[0.72, 0.28]),
            }
        )

df = pd.DataFrame(rows)

# Add a small amount of realistic missingness.
for col in ["quantity", "unit_price"]:
    idx = rng.choice(df.index, size=max(1, len(df) // 200), replace=False)
    df.loc[idx, col] = np.nan

Path("data").mkdir(exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print(f"Generated {len(df):,} transactions for {df['customer_id'].nunique():,} customers.")
print("Saved to data/transactions.csv")
