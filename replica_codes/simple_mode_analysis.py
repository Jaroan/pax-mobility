from google.cloud import bigquery
import pandas as pd

# ---------- SETTINGS ----------
PROJECT_DATASET = "replica-customer.nasa_ames"
DATASETS = [
    ("Q2 Thursday", "ca_2024_Q2_thursday_trip"),
    ("Q2 Saturday", "ca_2024_Q2_saturday_trip"),
    ("Q4 Thursday", "ca_2024_Q4_thursday_trip"),
    ("Q4 Saturday", "ca_2024_Q4_saturday_trip"),
]

client = bigquery.Client()

def get_simple_mode_counts(table_name: str) -> pd.DataFrame:
    """Get simple counts of trips by mode - this should show private_auto as #1"""
    query = f"""
    SELECT 
        mode,
        COUNT(*) AS total_trips,
        ROUND(AVG(distance_miles), 2) AS avg_distance_miles,
        ROUND(AVG(duration_minutes), 2) AS avg_duration_min,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
    FROM `{PROJECT_DATASET}.{table_name}`
    WHERE distance_miles > 0 AND duration_minutes > 0
    GROUP BY mode
    ORDER BY total_trips DESC
    """
    return client.query(query).to_dataframe()

def get_route_frequency_analysis(table_name: str, top_n: int = 10) -> pd.DataFrame:
    """Get most frequent exact routes (origin-destination pairs)"""
    query = f"""
    SELECT 
        mode,
        start_lat, start_lng, end_lat, end_lng,
        COUNT(*) AS route_frequency,
        ROUND(AVG(distance_miles), 2) AS avg_distance_miles,
        ROUND(AVG(duration_minutes), 2) AS avg_duration_min
    FROM `{PROJECT_DATASET}.{table_name}`
    WHERE distance_miles > 0 AND duration_minutes > 0
    GROUP BY mode, start_lat, start_lng, end_lat, end_lng
    ORDER BY route_frequency DESC
    LIMIT {top_n}
    """
    return client.query(query).to_dataframe()

print("🔍 SIMPLE MODE ANALYSIS - This should show private_auto as most popular")
print("=" * 80)

for dataset_label, table_name in DATASETS:
    print(f"\n📊 {dataset_label.upper()}")
    print("-" * 50)
    
    # Simple mode counts
    mode_counts = get_simple_mode_counts(table_name)
    print("MODE POPULARITY (by total trips):")
    print(f"{'Mode':<15} {'Total Trips':<15} {'Percentage':<12} {'Avg Distance':<12}")
    print("-" * 55)
    
    for _, row in mode_counts.head(5).iterrows():
        print(f"{row['mode']:<15} {int(row['total_trips']):>14,} {row['percentage']:>10.1f}% {row['avg_distance_miles']:>10.1f} mi")
    
    print(f"\n🔥 TOP 5 MOST FREQUENT EXACT ROUTES:")
    route_freq = get_route_frequency_analysis(table_name, 5)
    print(f"{'Mode':<12} {'Frequency':<10} {'Distance':<10} {'Duration':<10}")
    print("-" * 45)
    
    for _, row in route_freq.iterrows():
        print(f"{row['mode']:<12} {int(row['route_frequency']):>9,} {row['avg_distance_miles']:>8.1f} mi {row['avg_duration_min']:>8.1f} min")

print("\n" + "=" * 80)
print("💡 INTERPRETATION:")
print("• Mode Popularity = Total individual trips (private_auto should be #1)")
print("• Route Frequency = How often the exact same route is taken")
print("• Walking/transit might have higher route frequency (same stops/paths)")
print("• Private auto might have lower route frequency (more route variation)")
print("=" * 80)
