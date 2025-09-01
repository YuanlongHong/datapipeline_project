import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_parquet("yellow_tripdata_2025-07.parquet")

# Show basic information
print("=== NYC Taxi Data Analysis ===")
print(f"Dataset shape: {df.shape}")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Data Info ===")
print(df.info())

print("\n=== Statistical Summary ===")
print(df.describe(include="all"))

print("\n=== Missing Values ===")
missing_values = df.isna().sum()
print(missing_values)

print("\n=== Missing Values Summary ===")
total_missing = missing_values.sum()
total_cells = len(df) * len(df.columns)
print(f"Total missing values: {total_missing:,}")
print(f"Total cells: {total_cells:,}")
print(f"Missing percentage: {(total_missing/total_cells)*100:.2f}%")


# Integer/float columns - use -1
# df['passenger_count'] = df['passenger_count'].fillna(-1)
# df['RatecodeID'] = df['RatecodeID'].fillna(-1)
# df['congestion_surcharge'] = df['congestion_surcharge'].fillna(-1)
# df['Airport_fee'] = df['Airport_fee'].fillna(-1)

# # Text columns - use 'MISSING'
# df['store_and_fwd_flag'] = df['store_and_fwd_flag'].fillna('MISSING')



print("\n=== Simple Dirty Data Flagging ===")

# Step 1: Flag missing values (we already did this)
df['flag_missing_passenger'] = df['passenger_count'].isna()
df['flag_missing_ratecode'] = df['RatecodeID'].isna()
df['flag_missing_store_flag'] = df['store_and_fwd_flag'].isna()
df['flag_missing_congestion'] = df['congestion_surcharge'].isna()
df['flag_missing_airport_fee'] = df['Airport_fee'].isna()

# Step 2: Flag impossible values (obviously wrong)
df['flag_impossible_distance'] = (df['trip_distance'] <= 0)
df['flag_impossible_fare'] = (df['fare_amount'] <= 0)
df['flag_impossible_time'] = (df['tpep_pickup_datetime'] >= df['tpep_dropoff_datetime'])

# Step 3: Flag outliers using Z-score method
def flag_zscore_outliers(df, column, threshold=3):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return z_scores > threshold

df['flag_distance_zscore'] = flag_zscore_outliers(df, 'trip_distance', 3)
df['flag_fare_zscore'] = flag_zscore_outliers(df, 'fare_amount', 3)

# Step 4: Flag outliers using IQR method
def flag_iqr_outliers(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)

df['flag_distance_iqr'] = flag_iqr_outliers(df, 'trip_distance', 1.5)
df['flag_fare_iqr'] = flag_iqr_outliers(df, 'fare_amount', 1.5)

# Step 5: Show summary of all flags
print("�� Dirty Data Summary:")
print("=" * 40)

print("\n�� Missing Data Flags:")
print(f"   Missing passenger_count: {df['flag_missing_passenger'].sum():,}")
print(f"   Missing RatecodeID: {df['flag_missing_ratecode'].sum():,}")
print(f"   Missing store_and_fwd_flag: {df['flag_missing_store_flag'].sum():,}")
print(f"   Missing congestion_surcharge: {df['flag_missing_congestion'].sum():,}")
print(f"   Missing Airport_fee: {df['flag_missing_airport_fee'].sum():,}")

print("\n❌ Impossible Data Flags:")
print(f"   Impossible distance (≤0): {df['flag_impossible_distance'].sum():,}")
print(f"   Impossible fare (≤0): {df['flag_impossible_fare'].sum():,}")
print(f"   Impossible timing: {df['flag_impossible_time'].sum():,}")

print("\n📈 Outlier Flags (Z-score method):")
print(f"   Distance outliers (Z-score > 3): {df['flag_distance_zscore'].sum():,}")
print(f"   Fare outliers (Z-score > 3): {df['flag_fare_zscore'].sum():,}")

print("\n�� Outlier Flags (IQR method):")
print(f"   Distance outliers (IQR method): {df['flag_distance_iqr'].sum():,}")
print(f"   Fare outliers (IQR method): {df['flag_fare_iqr'].sum():,}")

# Step 6: Create overall dirty data flag
df['flag_any_dirty_data'] = (
    df['flag_missing_passenger'] | 
    df['flag_missing_ratecode'] | 
    df['flag_missing_store_flag'] | 
    df['flag_missing_congestion'] | 
    df['flag_missing_airport_fee'] |
    df['flag_impossible_distance'] | 
    df['flag_impossible_fare'] | 
    df['flag_impossible_time'] |
    df['flag_distance_zscore'] | 
    df['flag_fare_zscore'] |
    df['flag_distance_iqr'] | 
    df['flag_fare_iqr']
)

print(f"\n�� TOTAL RECORDS WITH ANY DIRTY DATA: {df['flag_any_dirty_data'].sum():,}")
print(f"�� CLEAN RECORDS: {(~df['flag_any_dirty_data']).sum():,}")




# Your existing code (lines 1-123)
# ... all your current code ...

# ===============================
# Step 1: Summary of Dirty Data
# ===============================
dirty_summary = pd.DataFrame({
    "Flag": [
        "Missing passenger_count",
        "Missing RatecodeID", 
        "Missing store_and_fwd_flag",
        "Missing congestion_surcharge",
        "Missing Airport_fee",
        "Impossible distance",
        "Impossible fare", 
        "Impossible time",
        "Distance outliers (Z-score)",
        "Fare outliers (Z-score)",
        "Distance outliers (IQR)",
        "Fare outliers (IQR)",
        "Any dirty data"
    ],
    "Count": [
        df['flag_missing_passenger'].sum(),
        df['flag_missing_ratecode'].sum(),
        df['flag_missing_store_flag'].sum(),
        df['flag_missing_congestion'].sum(),
        df['flag_missing_airport_fee'].sum(),
        df['flag_impossible_distance'].sum(),
        df['flag_impossible_fare'].sum(),
        df['flag_impossible_time'].sum(),
        df['flag_distance_zscore'].sum(),
        df['flag_fare_zscore'].sum(),
        df['flag_distance_iqr'].sum(),
        df['flag_fare_iqr'].sum(),
        df['flag_any_dirty_data'].sum()
    ]
})

print("=== Dirty Data Summary ===")
print(dirty_summary)

# ===============================
# Step 2: Visualization
# ===============================

# 1. Bar plot of dirty data counts
plt.figure(figsize=(12,6))
sns.barplot(x="Count", y="Flag", data=dirty_summary[:-1])  # skip 'Any dirty data'
plt.title("Dirty Data Counts by Type")
plt.xlabel("Number of Records")
plt.ylabel("Dirty Data Type")
plt.tight_layout()
plt.show()

# 2. Distribution of numerical fields (clean vs dirty)
numerical_cols = ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']

for col in numerical_cols:
    plt.figure(figsize=(10,4))
    sns.kdeplot(df[~df['flag_any_dirty_data']][col], label='Clean', fill=True)
    sns.kdeplot(df[df['flag_any_dirty_data']][col], label='Dirty', fill=True, alpha=0.5)
    plt.title(f"Distribution of {col} (Clean vs Dirty)")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===============================
# Step 3: Save Dirty and Clean Data
# ===============================
df[df['flag_any_dirty_data']].to_parquet("dirty_data.parquet")
df[~df['flag_any_dirty_data']].to_parquet("clean_data.parquet")

print("✅ Dirty and clean datasets saved!")