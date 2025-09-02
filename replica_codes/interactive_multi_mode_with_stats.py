from google.cloud import bigquery
import pandas as pd
import folium
from folium import plugins
from itertools import cycle

# ---------- SETTINGS ----------
PROJECT_DATASET = "replica-customer.nasa_ames"
DATASETS = [
    ("Q2 Thursday", "ca_2024_Q2_thursday_trip"),
    ("Q2 Saturday", "ca_2024_Q2_saturday_trip"),
    ("Q4 Thursday", "ca_2024_Q4_thursday_trip"),
    ("Q4 Saturday", "ca_2024_Q4_saturday_trip"),
]
PER_MODE_LIMIT = 1000       # how many OD flows per mode to draw (tune for performance)
INCLUDE_POP_JOIN = False    # set True only if you need demographics on-map
# -----------------------------

client = bigquery.Client()

# Some Replica projects use 'mode' values beyond these; colors recycle if needed.
PALETTE = [
    "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"
]

def fetch_flows(table_name: str) -> pd.DataFrame:
    # Build season prefix like 'ca_2024_Q2' (first 3 pieces)
    season_prefix = "_".join(table_name.split("_")[:3])
    population_table = f"{season_prefix}_population"

    join_sql = f"""
      JOIN `{PROJECT_DATASET}.{population_table}` AS pers
        ON pers.person_id = trip.person_id
    """ if INCLUDE_POP_JOIN else ""

    query = f"""
    -- Get per-mode top flows while keeping all modes visible
    WITH trips AS (
      SELECT
        trip.start_lat, trip.start_lng, trip.end_lat, trip.end_lng,
        trip.mode
      FROM `{PROJECT_DATASET}.{table_name}` AS trip
      {join_sql}
      WHERE
        -- keep only valid WGS84 coords
        trip.start_lat BETWEEN -90 AND 90  AND trip.start_lng BETWEEN -180 AND 180
        AND trip.end_lat   BETWEEN -90 AND 90  AND trip.end_lng   BETWEEN -180 AND 180
    ),
    agg AS (
      SELECT
        start_lat, start_lng, end_lat, end_lng, mode,
        COUNT(*) AS trip_count
      FROM trips
      GROUP BY start_lat, start_lng, end_lat, end_lng, mode
    )
    SELECT *
    FROM agg
    QUALIFY ROW_NUMBER() OVER (PARTITION BY mode ORDER BY trip_count DESC) <= {PER_MODE_LIMIT}
    ORDER BY mode, trip_count DESC
    """
    return client.query(query).to_dataframe()

def analyze_frequent_trips(table_name: str, top_n: int = 20) -> pd.DataFrame:
    """Analyze the most frequent origin-destination pairs with detailed statistics"""
    query = f"""
    WITH trip_aggregation AS (
      SELECT
        start_lat, start_lng, end_lat, end_lng,
        mode,
        COUNT(*) AS trip_count,
        ROUND(AVG(distance_miles), 2) AS avg_distance_miles,
        ROUND(AVG(duration_minutes), 2) AS avg_duration_min,
        MIN(distance_miles) AS min_distance_miles,
        MAX(distance_miles) AS max_distance_miles
      FROM `{PROJECT_DATASET}.{table_name}`
      WHERE
        start_lat BETWEEN -90 AND 90 AND start_lng BETWEEN -180 AND 180
        AND end_lat BETWEEN -90 AND 90 AND end_lng BETWEEN -180 AND 180
        AND distance_miles > 0 
        AND duration_minutes > 0
      GROUP BY start_lat, start_lng, end_lat, end_lng, mode
    )
    SELECT *
    FROM trip_aggregation
    ORDER BY trip_count DESC
    LIMIT {top_n}
    """
    return client.query(query).to_dataframe()

def get_mode_statistics(table_name: str) -> pd.DataFrame:
    """Get comprehensive statistics by mode"""
    query = f"""
    SELECT
      mode,
      COUNT(*) AS total_trips,
      ROUND(AVG(distance_miles), 2) AS avg_distance_miles,
      ROUND(AVG(duration_minutes), 2) AS avg_duration_min,
      ROUND(STDDEV(duration_minutes), 2) AS std_duration_min,
      APPROX_QUANTILES(duration_minutes, 4)[OFFSET(2)] AS median_duration_min,
      COUNT(DISTINCT CONCAT(start_lat, ',', start_lng, '-', end_lat, ',', end_lng)) AS unique_routes
    FROM `{PROJECT_DATASET}.{table_name}`
    WHERE distance_miles > 0 
      AND duration_minutes > 0
    GROUP BY mode
    ORDER BY total_trips DESC
    """
    return client.query(query).to_dataframe()


def create_statistics_panel(datasets_stats, frequent_trips_stats, mode_popularity_stats):
    """Create a comprehensive statistics panel for the HTML"""
    stats_html = """
    <div id="statsPanel" style="
     position: fixed; top: 10px; right: 10px; z-index: 9999;
     background: rgba(255, 255, 255, 0.95); padding: 15px; 
     border: 2px solid #333; border-radius: 8px;
     font: 11px/1.3 Arial, sans-serif; max-width: 350px;
     max-height: 80vh; overflow-y: auto;">
      <div style="text-align: center; margin-bottom: 10px;">
        <b>📊 Bay Area Trip Statistics</b>
        <button onclick="toggleStats()" style="float: right; font-size: 10px;">▼</button>
      </div>
      <div id="statsContent">
    """
    
    # Mode Popularity (what you expect to see)
    stats_html += "<b>� Mode Popularity (Total Trips):</b><br/>"
    for dataset_name, mode_pop in mode_popularity_stats.items():
        if not mode_pop.empty:
            top_mode = mode_pop.iloc[0]
            stats_html += f"""
            <details style="margin: 5px 0;">
              <summary><b>{dataset_name}</b></summary>
              <div style="margin-left: 10px; font-size: 10px;">
                • #1: {top_mode['mode']} ({top_mode['percentage']:.1f}%)<br/>
                • Total trips: {int(top_mode['total_individual_trips']):,}<br/>
                • Top 3: {', '.join(mode_pop.head(3)['mode'].tolist())}<br/>
              </div>
            </details>
            """
    
    # Route frequency analysis
    stats_html += "<br/><b>🔄 Route Frequency Analysis:</b><br/>"
    for dataset_name, stats in datasets_stats.items():
        stats_html += f"""
        <details style="margin: 5px 0;">
          <summary><b>{dataset_name} Routes</b></summary>
          <div style="margin-left: 10px; font-size: 10px;">
            • Total route instances: {stats['total_route_instances']:,}<br/>
            • Max route frequency: {stats['max_route_frequency']:,}<br/>
            • Avg route frequency: {stats['avg_route_frequency']:.1f}<br/>
          </div>
        </details>
        """
    
    # Top frequent trips
    stats_html += "<br/><b>🔥 Most Frequent Routes:</b><br/>"
    for dataset_name, freq_stats in frequent_trips_stats.items():
        if not freq_stats.empty:
            top_route = freq_stats.iloc[0]
            stats_html += f"""
            <details style="margin: 5px 0;">
              <summary><b>{dataset_name} Top Route</b></summary>
              <div style="margin-left: 10px; font-size: 10px;">
                • Mode: {top_route['mode']}<br/>
                • Frequency: {int(top_route['trip_count']):,}<br/>
                • Distance: {top_route['avg_distance_miles']:.1f} mi<br/>
                • Duration: {top_route['avg_duration_min']:.1f} min<br/>
              </div>
            </details>
            """
    
    stats_html += """
      </div>
    </div>
    
    <script>
    function toggleStats() {
      var content = document.getElementById('statsContent');
      var button = event.target;
      if (content.style.display === 'none') {
        content.style.display = 'block';
        button.innerHTML = '▼';
      } else {
        content.style.display = 'none';
        button.innerHTML = '▶';
      }
    }
    </script>
    """
    return stats_html


def add_top_routes_layer(m, df, frequent_trips_df, dataset_label, top_n=10):
    """Add a special layer highlighting the most frequent routes"""
    if frequent_trips_df.empty:
        return
        
    top_routes = frequent_trips_df.head(top_n)
    
    top_group = folium.FeatureGroup(
        name=f"🔥 {dataset_label} - Top {top_n} Routes", 
        show=False
    )
    top_group.add_to(m)
    
    for i, (_, route) in enumerate(top_routes.iterrows()):
        # Create enhanced popup with detailed statistics
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px; width: 200px;">
          <h4 style="margin: 0 0 10px 0; color: #d32f2f;">
            🏆 Rank #{i+1} Route
          </h4>
          <table style="width: 100%; font-size: 11px;">
            <tr><td><b>Dataset:</b></td><td>{dataset_label}</td></tr>
            <tr><td><b>Mode:</b></td><td>{route['mode']}</td></tr>
            <tr><td><b>Trip Count:</b></td><td>{int(route['trip_count']):,}</td></tr>
            <tr><td><b>Avg Distance:</b></td><td>{route['avg_distance_miles']:.1f} mi</td></tr>
            <tr><td><b>Avg Duration:</b></td><td>{route['avg_duration_min']:.1f} min</td></tr>
            <tr><td><b>Distance Range:</b></td><td>{route['min_distance_miles']:.1f} - {route['max_distance_miles']:.1f} mi</td></tr>
          </table>
        </div>
        """
        
        # Create polyline with special styling for top routes
        folium.PolyLine(
            locations=[(route['start_lat'], route['start_lng']),
                      (route['end_lat'], route['end_lng'])],
            color='#d32f2f',  # Red color for top routes
            weight=6 + (top_n - i) * 0.5,  # Thicker lines for higher ranks
            opacity=0.9,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"#{i+1}: {int(route['trip_count']):,} trips ({route['mode']})"
        ).add_to(top_group)
        
        # Add start and end markers for top routes
        folium.CircleMarker(
            location=(route['start_lat'], route['start_lng']),
            radius=8,
            popup=f"Start: Rank #{i+1}",
            color='green',
            fill=True,
            fillOpacity=0.7
        ).add_to(top_group)
        
        folium.CircleMarker(
            location=(route['end_lat'], route['end_lng']),
            radius=8,
            popup=f"End: Rank #{i+1}",
            color='red',
            fill=True,
            fillOpacity=0.7
        ).add_to(top_group)


def print_trip_statistics(datasets_stats, frequent_trips_stats, mode_stats, mode_popularity_stats):
    """Print comprehensive trip statistics to console"""
    print("
" + "="*80)
    print("🚀 BAY AREA TRIP ANALYSIS REPORT")
    print("="*80)
    
    # Mode Popularity (this should show private_auto as #1)
    print("
🚗 MODE POPULARITY (Total Individual Trips)")
    print("-" * 80)
    
    for dataset_name, mode_pop in mode_popularity_stats.items():
        if not mode_pop.empty:
            print(f"
{dataset_name}:")
            print(f"{'Mode':<15} {'Total Trips':<15} {'Percentage':<12}")
            print("-" * 45)
            
            for _, row in mode_pop.head(5).iterrows():
                print(f"{row['mode']:<15} {int(row['total_individual_trips']):>14,} {row['percentage']:>10.1f}%")
    
    # Route frequency analysis
    print("
� ROUTE FREQUENCY ANALYSIS (Repeated Routes)")
    print("-" * 80)
    print(f"{'Dataset':<20} {'Route Instances':<15} {'Max Frequency':<12} {'Avg Frequency':<12}")
    print("-" * 65)
    
    for dataset_name, stats in datasets_stats.items():
        print(f"{dataset_name:<20} {stats['total_route_instances']:>14,} {stats['max_route_frequency']:>11,} {stats['avg_route_frequency']:>11.1f}")
    
    # Most frequent trips
    print("
🔥 MOST FREQUENT EXACT ROUTES BY DATASET")
    print("-" * 80)
    
    for dataset_name, freq_df in frequent_trips_stats.items():
        if not freq_df.empty:
            print(f"
{dataset_name}:")
            print(f"{'Rank':<5} {'Mode':<12} {'Frequency':<10} {'Dist(mi)':<10} {'Dur(min)':<10}")
            print("-" * 50)
            
            for i, (_, row) in enumerate(freq_df.head(5).iterrows()):
                print(f"{i+1:<5} {row['mode']:<12} {int(row['trip_count']):<10} "
                      f"{row['avg_distance_miles']:<10.1f} {row['avg_duration_min']:<10.1f}")

# Print statistics to console
print_trip_statistics(datasets_stats, frequent_trips_stats, mode_stats, mode_popularity_stats)


# Create one map and add dataset groups, then mode subgroups
m = folium.Map(location=[37.77, -122.42], zoom_start=10, tiles="CartoDB positron")

# Helpful plugins
plugins.Fullscreen().add_to(m)
plugins.MiniMap(toggle_display=True).add_to(m)
plugins.MousePosition(position="bottomleft", prefix="Lat/Lng").add_to(m)

# Collect statistics for all datasets
print("📊 Analyzing trip patterns across all datasets...")
datasets_stats = {}
frequent_trips_stats = {}
mode_stats = {}
mode_popularity_stats = {}

for dataset_label, table_name in DATASETS:
    print(f"Processing {dataset_label}...")
    
    # Get flow data (route frequency analysis)
    df = fetch_flows(table_name)
    if df.empty:
        continue
    
    # Calculate ROUTE frequency statistics (from flow data)
    datasets_stats[dataset_label] = {
        'total_route_instances': df['trip_count'].sum(),
        'max_route_frequency': df['trip_count'].max(),
        'avg_route_frequency': df['trip_count'].mean(),
        'top_route_modes': df.groupby('mode')['trip_count'].sum().nlargest(3).index.tolist()
    }
    
    # Get MODE popularity (total individual trips by mode)
    mode_popularity_query = f"""
    SELECT 
        mode,
        COUNT(*) AS total_individual_trips,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS percentage
    FROM `{PROJECT_DATASET}.{table_name}`
    WHERE distance_miles > 0 AND duration_minutes > 0
    GROUP BY mode
    ORDER BY total_individual_trips DESC
    """
    mode_popularity_stats[dataset_label] = client.query(mode_popularity_query).to_dataframe()
    
    # Get detailed frequent trips analysis
    frequent_trips_stats[dataset_label] = analyze_frequent_trips(table_name, top_n=20)
    
    # Get mode statistics
    mode_stats[dataset_label] = get_mode_statistics(table_name)

# Print statistics to console
print_trip_statistics(datasets_stats, frequent_trips_stats, mode_stats)

# Now create the map layers
for dataset_label, table_name in DATASETS:
    df = fetch_flows(table_name)
    if df.empty:
        continue

    # Parent layer per dataset (start hidden to avoid heavy initial render)
    parent_group = folium.FeatureGroup(name=f"{dataset_label}", show=False)
    parent_group.add_to(m)

    # Add top routes layer for this dataset
    if dataset_label in frequent_trips_stats:
        add_top_routes_layer(m, df, frequent_trips_stats[dataset_label], dataset_label)

    # Color map per mode in this dataset
    modes = list(pd.Series(df["mode"].astype(str).fillna("UNKNOWN")).unique())
    color_cycle = cycle(PALETTE)
    color_by_mode = {mode: (next(color_cycle)) for mode in modes}

    # Subgroup per mode
    for mode in modes:
        sub = df[df["mode"] == mode]
        sub_group = plugins.FeatureGroupSubGroup(parent_group, name=f"{dataset_label} — {mode}")
        sub_group.add_to(m)

        # Draw flows as polylines with width ~ trip_count
        for _, r in sub.iterrows():
            # Enhanced tooltip with rank information
            rank = sub.index.get_loc(r.name) + 1
            enhanced_tooltip = (f"{dataset_label} | {mode} | "
                              f"trips: {int(r.trip_count)} | rank: #{rank}")
            
            folium.PolyLine(
                locations=[(float(r.start_lat), float(r.start_lng)),
                           (float(r.end_lat),   float(r.end_lng))],
                color=color_by_mode[mode],
                weight=1 + (float(r.trip_count) ** 0.30),  # gentle scaling
                opacity=0.6,
                tooltip=enhanced_tooltip
            ).add_to(sub_group)

        # Optional: clustered start points for this mode (quick density feel)
        cluster = plugins.MarkerCluster(name=f"{dataset_label} — {mode} (origin clusters)")
        for _, r in sub.sample(min(len(sub), 2000), random_state=0).iterrows():
            folium.CircleMarker(
                location=(float(r.start_lat), float(r.start_lng)),
                radius=2,
                fill=True, opacity=0.5, fill_opacity=0.5,
                color=color_by_mode[mode]
            ).add_to(cluster)
        cluster.add_to(sub_group)

# Grouped control makes the menu cleaner; regular LayerControl also works
try:
    plugins.GroupedLayerControl().add_to(m)
except Exception:
    folium.LayerControl(collapsed=False).add_to(m)

# Add comprehensive statistics panel
stats_panel = create_statistics_panel(datasets_stats, frequent_trips_stats, mode_popularity_stats)
m.get_root().html.add_child(folium.Element(stats_panel))

# Enhanced legend with trip statistics information
legend_html = """
<div style="
 position: fixed; bottom: 10px; left: 10px; z-index: 9999;
 background: white; padding: 10px 12px; border: 1px solid #999; border-radius: 6px;
 font: 12px/1.2 Arial, sans-serif; max-width: 300px;">
  <b>🗺️ Bay Area Trip Flow Map</b><br/>
  
  <b>📈 Visual Elements:</b><br/>
  • Line thickness = Trip frequency<br/>
  • 🔥 Red lines = Top routes layer<br/>
  • Colors = Transport modes<br/>
  • Clusters = Origin points<br/><br/>
  
  <b>🎛️ Interactive Controls:</b><br/>
  • Toggle datasets (Q2/Q4, Thu/Sat)<br/>
  • Toggle transport modes<br/>
  • View top routes highlights<br/>
  • Check statistics panel (top-right)<br/><br/>
  
  <b>💡 Usage Tips:</b><br/>
  • Hover lines for trip details<br/>
  • Click top routes for statistics<br/>
  • Use fullscreen for better view<br/>
  • Statistics show frequent patterns<br/>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Save the enhanced map
output_file = "replica_bayarea_interactive_flows_with_stats.html"
m.save(output_file)
print(f"\n🎉 Saved enhanced interactive map: {output_file}")
print(f"📊 Map includes statistics for {len(datasets_stats)} datasets")
print(f"🔥 Highlighting top frequent routes for each dataset")
print(f"📈 Comprehensive trip analysis completed!")

# Summary of what was created
print("\n" + "="*60)
print("📋 GENERATED FEATURES SUMMARY")
print("="*60)
print("✅ Interactive map with trip flow visualization")
print("✅ Statistics panel (collapsible, top-right)")
print("✅ Top routes highlighting layer")
print("✅ Enhanced tooltips with rank information")
print("✅ Console output with detailed statistics")
print("✅ Mode-wise and dataset-wise comparisons")
print("✅ Distance and duration analysis")
print("="*60)
