import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime
import uuid
import os
import random
from google.cloud import storage

# ====================== CONFIG ======================
BUCKET_NAME = "ecommerce-data-drift-detection"          # ← CHANGE THIS
GCS_RAW_PREFIX = "raw/orders/"
# ====================================================

fake = Faker()


def generate_ecommerce_data(
    num_records: int = 5000,
    inject_schema_change: bool = False,
    inject_data_drift: bool = False,
    inject_nulls: bool = False,
    inject_duplicates: bool = False,
    drift_strength: float = 0.3
) -> pd.DataFrame:
    """
    Generate realistic e-commerce data with optional schema changes and data drift.
    """

    print("🚀 Generating base data...")

    data = {
        "order_id": [str(uuid.uuid4()) for _ in range(num_records)],
        "customer_id": [f"CUST_{fake.random_int(10000, 99999)}" for _ in range(num_records)],
        "order_date": [fake.date_time_between(start_date="-30d", end_date="now") for _ in range(num_records)],
        "product_id": [f"PROD_{fake.random_int(1000, 9999)}" for _ in range(num_records)],
        "product_name": [fake.word().capitalize() + " " + fake.word().capitalize() for _ in range(num_records)],
        "category": np.random.choice(
            ["Electronics", "Fashion", "Home", "Beauty", "Sports", "Books"], num_records
        ),
        "quantity": np.random.randint(1, 5, num_records),
        "unit_price": np.round(np.random.uniform(10, 500, num_records), 2),
        "discount_percent": np.round(np.random.uniform(0, 30, num_records), 1),
        "payment_method": np.random.choice(
            ["Credit Card", "Debit Card", "UPI", "COD", "Net Banking", "Wallet"],
            num_records, p=[0.30, 0.15, 0.25, 0.15, 0.10, 0.05]
        ),
        "order_status": np.random.choice(
            ["Pending", "Confirmed", "Shipped", "Delivered", "Cancelled", "Returned"],
            num_records, p=[0.10, 0.15, 0.20, 0.40, 0.10, 0.05]
        ),
        "shipping_city": [fake.city() for _ in range(num_records)],
        "shipping_state": [fake.state() for _ in range(num_records)],
        "shipping_pincode": [fake.postcode() for _ in range(num_records)],
        "customer_email": [fake.email() for _ in range(num_records)],
        "is_first_order": np.random.choice([True, False], num_records, p=[0.30, 0.70]),
        "device_type": np.random.choice(["Mobile", "Desktop", "Tablet"], num_records, p=[0.65, 0.30, 0.05]),
        "delivery_partner": np.random.choice(
            ["Delhivery", "BlueDart", "Ekart", "Ecom Express", "XpressBees"], num_records
        ),
        "customer_segment": np.random.choice(
            ["New", "Regular", "Loyal", "VIP"], num_records, p=[0.30, 0.40, 0.20, 0.10]
        ),
        "order_source": np.random.choice(["App", "Website", "M-Site"], num_records, p=[0.55, 0.35, 0.10]),
        "is_prime_member": np.random.choice([True, False], num_records, p=[0.25, 0.75]),
        "estimated_delivery_days": np.random.randint(1, 8, num_records),
    }

    df = pd.DataFrame(data)

    # Derived columns
    df["gross_amount"] = np.round(df["quantity"] * df["unit_price"], 2)
    df["discount_amount"] = np.round(df["gross_amount"] * df["discount_percent"] / 100, 2)
    df["net_amount"] = np.round(df["gross_amount"] - df["discount_amount"], 2)
    df["tax_amount"] = np.round(df["net_amount"] * 0.18, 2)
    df["total_amount"] = np.round(df["net_amount"] + df["tax_amount"], 2)
    df["actual_delivery_days"] = df["estimated_delivery_days"] + np.random.randint(-1, 3, num_records)
    df["rating"] = np.random.choice([1, 2, 3, 4, 5, None], num_records, p=[0.05, 0.05, 0.10, 0.25, 0.35, 0.20])
    df["review_present"] = df["rating"].notna()

    # Metadata
    df["ingestion_timestamp"] = datetime.now()
    df["source_system"] = "order_service"
    df["batch_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"✅ Base data created: {df.shape[0]} rows, {df.shape[1]} columns")

    # ========== APPLY OPTIONAL BEHAVIORS ==========

    if inject_schema_change:
        df = _apply_schema_change(df)

    if inject_data_drift:
        df = _apply_data_drift(df, strength=drift_strength)

    if inject_nulls:
        df = _inject_nulls(df)

    if inject_duplicates:
        df = _inject_duplicates(df)

    print(f"📊 Final data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def _apply_schema_change(df: pd.DataFrame) -> pd.DataFrame:
    """Add new columns randomly to simulate schema evolution"""
    possible_new_columns = {
        "delivery_rating": lambda: np.random.randint(1, 6, len(df)),
        "payment_gateway": lambda: np.random.choice(
            ["Razorpay", "Paytm", "Stripe", "PhonePe", "Cashfree"], len(df)
        ),
        "is_express_delivery": lambda: np.random.choice([True, False], len(df), p=[0.2, 0.8]),
        "warehouse_id": lambda: [f"WH_{fake.random_int(100, 999)}" for _ in range(len(df))],
        "coupon_used": lambda: np.random.choice([True, False], len(df), p=[0.35, 0.65]),
        "browser": lambda: np.random.choice(["Chrome", "Safari", "Firefox", "Edge"], len(df)),
        "app_version": lambda: [
            f"{random.randint(3,5)}.{random.randint(0,9)}.{random.randint(0,9)}" for _ in range(len(df))
        ],
    }

    num_new_cols = random.randint(1, 2)
    selected_cols = random.sample(list(possible_new_columns.keys()), num_new_cols)

    for col in selected_cols:
        df[col] = possible_new_columns[col]()
        print(f"🆕 Schema Change → Added column: {col}")

    return df


def _apply_data_drift(df: pd.DataFrame, strength: float = 0.3) -> pd.DataFrame:
    """Apply data drift by changing distributions"""
    print(f"📉 Applying data drift (strength={strength})")

    if random.random() < 0.7:
        mask = np.random.rand(len(df)) < strength
        df.loc[mask, "payment_method"] = "COD"
        print("   → Increased COD payments")

    if random.random() < 0.7:
        df["unit_price"] = np.round(df["unit_price"] * np.random.uniform(1.15, 1.5), 2)
        print("   → Increased unit prices")

    if random.random() < 0.6:
        mask = np.random.rand(len(df)) < strength
        df.loc[mask, "device_type"] = "Mobile"
        print("   → Increased Mobile orders")

    if random.random() < 0.5:
        mask = np.random.rand(len(df)) < (strength * 0.6)
        df.loc[mask, "order_status"] = "Cancelled"
        print("   → Increased cancellations")

    # Recalculate amounts
    df["gross_amount"] = np.round(df["quantity"] * df["unit_price"], 2)
    df["discount_amount"] = np.round(df["gross_amount"] * df["discount_percent"] / 100, 2)
    df["net_amount"] = np.round(df["gross_amount"] - df["discount_amount"], 2)
    df["tax_amount"] = np.round(df["net_amount"] * 0.18, 2)
    df["total_amount"] = np.round(df["net_amount"] + df["tax_amount"], 2)

    return df


def _inject_nulls(df: pd.DataFrame, null_percentage: float = 0.05) -> pd.DataFrame:
    cols = ["customer_email", "delivery_partner", "rating"]
    for col in cols:
        if col in df.columns:
            mask = np.random.rand(len(df)) < null_percentage
            df.loc[mask, col] = None
    print("⚠️  Injected null values")
    return df


def _inject_duplicates(df: pd.DataFrame, duplicate_percentage: float = 0.03) -> pd.DataFrame:
    n = int(len(df) * duplicate_percentage)
    duplicates = df.sample(n)
    df = pd.concat([df, duplicates], ignore_index=True)
    print(f"⚠️  Injected {n} duplicate records")
    return df


def upload_to_gcs(df: pd.DataFrame, bucket_name: str, gcs_prefix: str) -> str:
    """Upload DataFrame as Parquet to GCS"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"orders_{timestamp}.parquet"
    local_path = f"/tmp/{filename}" if os.name != "nt" else f"./{filename}"

    df.to_parquet(local_path, index=False)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{gcs_prefix}{filename}")
    blob.upload_from_filename(local_path)

    if os.path.exists(local_path):
        os.remove(local_path)

    gcs_uri = f"gs://{bucket_name}/{gcs_prefix}{filename}"
    print(f"✅ Uploaded → {gcs_uri}")
    return gcs_uri


def main():
    # ================== CONTROL PANEL ==================
    df = generate_ecommerce_data(
        num_records=5000,
        inject_schema_change=True,      # Add new columns
        inject_data_drift=True,         # Change distributions
        inject_nulls=True,              # Add nulls
        inject_duplicates=True,         # Add duplicates
        drift_strength=0.35             # Drift intensity
    )
    # ===================================================

    upload_to_gcs(df, BUCKET_NAME, GCS_RAW_PREFIX)
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()