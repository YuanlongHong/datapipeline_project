import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Load Data and Basic Info
# ===============================
print("=== NYC Taxi Data Analysis ===")

# Load data
df = pd.read_parquet("yellow_tripdata_2025-07.parquet")

# Show basic info
print(f"Dataset shape: {df.shape}")
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Show first few rows
print("\n=== First 5 rows ===")
print(df.head())

# Data info
print("\n=== Data Info ===")
print(df.info())

# Statistical summary
print("\n=== Statistical Summary ===")
print(df.describe(include="all"))

# ===============================
# 2. Missing Values Analysis
# ===============================
print("\n=== Missing Values Analysis ===")
missing_values = df.isna().sum()
print("Missing values by column:")
print(missing_values)

# Calculate missing percentage
total_missing = missing_values.sum()
total_cells = len(df) * len(df.columns)
print(f"\nMissing values summary:")
print(f"Total missing: {total_missing:,}")
print(f"Total cells: {total_cells:,}")
print(f"Missing percentage: {(total_missing/total_cells)*100:.2f}%")

# ===============================
# 3. Data Quality Flagging
# ===============================
print("\n=== Creating Data Quality Flags ===")

# 3.1 Missing value flags
print("Flagging missing values...")
df['flag_missing_passenger'] = df['passenger_count'].isna()
df['flag_missing_ratecode'] = df['RatecodeID'].isna()
df['flag_missing_store_flag'] = df['store_and_fwd_flag'].isna()
df['flag_missing_congestion'] = df['congestion_surcharge'].isna()
df['flag_missing_airport_fee'] = df['Airport_fee'].isna()

# 3.2 Impossible value flags
print("Flagging impossible values...")
df['flag_impossible_distance'] = (df['trip_distance'] <= 0)
df['flag_impossible_fare'] = (df['fare_amount'] <= 0)
df['flag_impossible_time'] = (df['tpep_pickup_datetime'] >= df['tpep_dropoff_datetime'])

# 3.3 Outlier detection - Z-score method
print("Flagging outliers using Z-score method...")
def flag_zscore_outliers(data, column, threshold=3):
    """Flag outliers using Z-score method"""
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    return z_scores > threshold

df['flag_distance_zscore'] = flag_zscore_outliers(df, 'trip_distance', 3)
df['flag_fare_zscore'] = flag_zscore_outliers(df, 'fare_amount', 3)

# 3.4 Outlier detection - IQR method
print("Flagging outliers using IQR method...")
def flag_iqr_outliers(data, column, multiplier=1.5):
    """Flag outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

df['flag_distance_iqr'] = flag_iqr_outliers(df, 'trip_distance', 1.5)
df['flag_fare_iqr'] = flag_iqr_outliers(df, 'fare_amount', 1.5)

# 3.5 Overall dirty data flag
print("Creating overall dirty data flag...")
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

# ===============================
# 4. Data Quality Report
# ===============================
print("\n=== Data Quality Report ===")
print("=" * 50)

print("\nMissing Value Flags:")
print(f"   passenger_count missing: {df['flag_missing_passenger'].sum():,}")
print(f"   RatecodeID missing: {df['flag_missing_ratecode'].sum():,}")
print(f"   store_and_fwd_flag missing: {df['flag_missing_store_flag'].sum():,}")
print(f"   congestion_surcharge missing: {df['flag_missing_congestion'].sum():,}")
print(f"   Airport_fee missing: {df['flag_missing_airport_fee'].sum():,}")

print("\nImpossible Value Flags:")
print(f"   Distance ≤ 0: {df['flag_impossible_distance'].sum():,}")
print(f"   Fare ≤ 0: {df['flag_impossible_fare'].sum():,}")
print(f"   Invalid timing: {df['flag_impossible_time'].sum():,}")

print("\nOutlier Flags (Z-score method):")
print(f"   Distance outliers: {df['flag_distance_zscore'].sum():,}")
print(f"   Fare outliers: {df['flag_fare_zscore'].sum():,}")

print("\nOutlier Flags (IQR method):")
print(f"   Distance outliers: {df['flag_distance_iqr'].sum():,}")
print(f"   Fare outliers: {df['flag_fare_iqr'].sum():,}")

print(f"\nOverall Summary:")
print(f"   Records with issues: {df['flag_any_dirty_data'].sum():,}")
print(f"   Clean records: {(~df['flag_any_dirty_data']).sum():,}")

# ===============================
# 5. Data Quality Summary Table
# ===============================
print("\n=== Data Quality Summary Table ===")

# Create summary DataFrame
dirty_summary = pd.DataFrame({
    "Issue Type": [
        "Missing passenger_count",
        "Missing RatecodeID", 
        "Missing store_and_fwd_flag",
        "Missing congestion_surcharge",
        "Missing Airport_fee",
        "Invalid distance",
        "Invalid fare", 
        "Invalid timing",
        "Distance outliers (Z-score)",
        "Fare outliers (Z-score)",
        "Distance outliers (IQR)",
        "Fare outliers (IQR)",
        "Any issue"
    ],
    "Record Count": [
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

print(dirty_summary)

# ===============================
# 6. Professional Visualizations
# ===============================
print("\n=== Creating Professional Visualizations ===")

# Set style for professional look
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 6.1 Clean Bar Chart of Data Issues
plt.figure(figsize=(14, 8))
ax = sns.barplot(x="Record Count", y="Issue Type", data=dirty_summary[:-1], palette="viridis")
plt.title("Data Quality Issues by Type", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Number of Records", fontsize=12)
plt.ylabel("Issue Type", fontsize=12)

# Add value labels on bars
for i, v in enumerate(dirty_summary[:-1]['Record Count']):
    ax.text(v + v*0.01, i, f'{v:,}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# 6.2 Clean vs Dirty Data Distribution (with clipping and log scale)
print("Creating distribution comparison plots...")

# Define columns to analyze
numerical_cols = ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']

for col in numerical_cols:
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get clean and dirty data
    clean_data = df[~df['flag_any_dirty_data']][col]
    dirty_data = df[df['flag_any_dirty_data']][col]
    
    # Plot 1: Clipped distribution (clean and readable)
    # Clip at 95th percentile for better visualization
    clip_value = clean_data.quantile(0.95)
    
    # Clean data (clipped)
    clean_clipped = clean_data[clean_data <= clip_value]
    ax1.hist(clean_clipped, bins=50, alpha=0.7, color='#2E8B57', label=f'Clean Data (≤{clip_value:.1f})', density=True)
    
    # Dirty data (clipped)
    dirty_clipped = dirty_data[dirty_data <= clip_value]
    ax1.hist(dirty_clipped, bins=50, alpha=0.7, color='#CD5C5C', label=f'Dirty Data (≤{clip_value:.1f})', density=True)
    
    ax1.set_title(f'{col} Distribution (Clipped at 95th Percentile)', fontweight='bold')
    ax1.set_xlabel(col)
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale for full range
    # Use log scale to handle wide range
    ax2.hist(clean_data, bins=50, alpha=0.7, color='#2E8B57', label='Clean Data', density=True, log=True)
    ax2.hist(dirty_data, bins=50, alpha=0.7, color='#CD5C5C', label='Dirty Data', density=True, log=True)
    
    ax2.set_title(f'{col} Distribution (Log Scale)', fontweight='bold')
    ax2.set_xlabel(col)
    ax2.set_ylabel('Density (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 6.3 Professional Summary Dashboard
print("Creating summary dashboard...")

# Create a summary figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Data quality overview
clean_count = (~df['flag_any_dirty_data']).sum()
dirty_count = df['flag_any_dirty_data'].sum()
colors = ['#2E8B57', '#CD5C5C']
ax1.pie([clean_count, dirty_count], labels=['Clean Data', 'Dirty Data'], 
         colors=colors, autopct='%1.1f%%', startangle=90)
ax1.set_title('Data Quality Overview', fontweight='bold', fontsize=14)

# Plot 2: Missing values breakdown
missing_cols = ['passenger_count', 'RatecodeID', 'store_and_fwd_flag', 'congestion_surcharge', 'Airport_fee']
missing_counts = [df[col].isna().sum() for col in missing_cols]
ax2.barh(missing_cols, missing_counts, color='#FF6B6B')
ax2.set_title('Missing Values by Column', fontweight='bold', fontsize=14)
ax2.set_xlabel('Count')

# Plot 3: Outlier comparison (Z-score vs IQR)
zscore_outliers = df['flag_distance_zscore'].sum() + df['flag_fare_zscore'].sum()
iqr_outliers = df['flag_distance_iqr'].sum() + df['flag_fare_iqr'].sum()
ax3.bar(['Z-score Method', 'IQR Method'], [zscore_outliers, iqr_outliers], 
         color=['#4ECDC4', '#45B7D1'])
ax3.set_title('Outlier Detection Comparison', fontweight='bold', fontsize=14)
ax3.set_ylabel('Number of Outliers')

# Plot 4: Data quality score
quality_score = (clean_count / len(df)) * 100
ax4.bar(['Data Quality Score'], [quality_score], color='#96CEB4')
ax4.set_ylim(0, 100)
ax4.set_title('Overall Data Quality Score', fontweight='bold', fontsize=14)
ax4.set_ylabel('Percentage (%)')
ax4.text(0, quality_score + 2, f'{quality_score:.1f}%', ha='center', fontweight='bold', fontsize=16)

plt.tight_layout()
plt.show()

# ===============================
# 7. Save Processed Data
# ===============================
print("\n=== Saving Processed Data ===")

# Save dirty data
dirty_data = df[df['flag_any_dirty_data']]
dirty_data.to_parquet("dirty_data.parquet", index=False)
print(f"✅ Dirty data saved: {len(dirty_data):,} records")

# Save clean data
clean_data = df[~df['flag_any_dirty_data']]
clean_data.to_parquet("clean_data.parquet", index=False)
print(f"✅ Clean data saved: {len(clean_data):,} records")

print("\n🎉 Data analysis complete!")
print(f"Original data: {len(df):,} records")
print(f"Clean data: {len(clean_data):,} records")
print(f"Dirty data: {len(dirty_data):,} records")