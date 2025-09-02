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

def fetch_flows_with_network_segments(table_name: str) -> pd.DataFrame:
    """Get trips with actual network segment trajectories"""
    # Determine which network segments table to use based on the trip table
    if "Q2" in table_name:
        network_table = "ca_2024_Q2_network_segments"
    elif "Q4" in table_name:
        network_table = "ca_2024_Q4_network_segments"
    else:
        network_table = "ca_2024_Q4_network_segments"  # fallback
    
    # First, let's check the schema of the network segments table
    schema_query = f"""
    SELECT column_name, data_type 
    FROM `{PROJECT_DATASET}.INFORMATION_SCHEMA.COLUMNS` 
    WHERE table_name = '{network_table.split('.')[-1]}'
    ORDER BY column_name
    """
    
    print(f"🔍 Checking schema for {network_table}...")
    try:
        schema_df = client.query(schema_query).to_dataframe()
        print("Available columns in network segments table:")
        for _, row in schema_df.iterrows():
            print(f"  - {row['column_name']} ({row['data_type']})")
    except Exception as e:
        print(f"Could not get schema: {e}")
    
    query = f"""
    -- Get trips from Oakley to Palo Alto with network trajectories
    WITH oakley_palo_alto_trips AS (
      SELECT
        trip.start_lat, trip.start_lng, trip.end_lat, trip.end_lng,
        trip.mode, trip.distance_miles, trip.duration_minutes,
        trip.network_link_ids,
        -- Create a unique identifier for each trip
        ROW_NUMBER() OVER (ORDER BY trip.start_lat, trip.start_lng, trip.mode) AS trip_row_id
      FROM `{PROJECT_DATASET}.{table_name}` AS trip
      WHERE
        -- keep only valid WGS84 coords
        trip.start_lat BETWEEN -90 AND 90  AND trip.start_lng BETWEEN -180 AND 180
        AND trip.end_lat   BETWEEN -90 AND 90  AND trip.end_lng   BETWEEN -180 AND 180
        -- Oakley area (Contra Costa County) - approximate bounding box
        AND trip.start_lat BETWEEN 37.98 AND 38.02
        AND trip.start_lng BETWEEN -121.73 AND -121.67
        -- Palo Alto area (San Mateo County) - approximate bounding box  
        AND trip.end_lat BETWEEN 37.39 AND 37.47
        AND trip.end_lng BETWEEN -122.18 AND -122.10
        AND trip.network_link_ids IS NOT NULL
        AND ARRAY_LENGTH(trip.network_link_ids) > 0
    ),
    trip_trajectories AS (
      SELECT
        t.trip_row_id,
        t.start_lat, t.start_lng, t.end_lat, t.end_lng,
        t.mode, t.distance_miles, t.duration_minutes,
        -- Create complete trajectory for each individual trip
        STRING_AGG(ST_AsText(ns.geometry), '|' ORDER BY link_position) AS trajectory_segments,
        -- Count how many segments were successfully matched
        COUNT(ns.geometry) AS matched_segments,
        -- Count total network links in the trip
        COUNT(link_id) AS total_links
      FROM oakley_palo_alto_trips t,
      UNNEST(t.network_link_ids) AS link_id WITH OFFSET AS link_position
      LEFT JOIN `{PROJECT_DATASET}.{network_table}` ns
        ON CAST(link_id AS STRING) = CAST(ns.stableEdgeId AS STRING)
      GROUP BY t.trip_row_id, t.start_lat, t.start_lng, t.end_lat, t.end_lng, 
               t.mode, t.distance_miles, t.duration_minutes
      -- Only include trips where at least 50% of segments were matched
      HAVING matched_segments > 0 AND (matched_segments / total_links) >= 0.5
    ),
    route_patterns AS (
      SELECT
        start_lat, start_lng, end_lat, end_lng, mode,
        trajectory_segments,
        COUNT(*) AS trip_count,
        ROUND(AVG(distance_miles), 2) AS avg_distance_miles,
        ROUND(AVG(duration_minutes), 2) AS avg_duration_min
      FROM trip_trajectories
      GROUP BY start_lat, start_lng, end_lat, end_lng, mode, trajectory_segments
      HAVING COUNT(*) >= 1  -- Only show routes with at least 1 trip
    )
    SELECT *
    FROM route_patterns
    ORDER BY trip_count DESC
    """
    return client.query(query).to_dataframe()

def parse_trajectory_coordinates(trajectory_segments):
    """Parse WKT LINESTRING geometries and extract coordinates in sequence"""
    if pd.isna(trajectory_segments) or not trajectory_segments:
        return []
    
    all_coordinates = []
    segments = trajectory_segments.split('|')
    
    for segment in segments:
        if 'LINESTRING' in segment:
            # Extract coordinates from LINESTRING format
            coords_str = segment.replace('LINESTRING (', '').replace(')', '')
            coord_pairs = coords_str.split(', ')
            
            segment_coords = []
            for pair in coord_pairs:
                try:
                    lng, lat = pair.split(' ')
                    segment_coords.append([float(lat), float(lng)])
                except (ValueError, IndexError):
                    continue
            
            # Connect segments: if this isn't the first segment, skip the first point 
            # to avoid duplication at intersections
            if all_coordinates and segment_coords:
                # Check if the first point of this segment is close to the last point of previous segment
                last_point = all_coordinates[-1]
                first_point = segment_coords[0]
                
                # If points are very close (within ~50 meters), skip the duplicate
                if (abs(last_point[0] - first_point[0]) < 0.0005 and 
                    abs(last_point[1] - first_point[1]) < 0.0005):
                    all_coordinates.extend(segment_coords[1:])  # Skip first point
                else:
                    all_coordinates.extend(segment_coords)  # Add all points
            else:
                all_coordinates.extend(segment_coords)
    
    return all_coordinates

def fetch_oakley_palo_alto_details(table_name: str) -> pd.DataFrame:
    """Get detailed trip information for Oakley to Palo Alto trips"""
    query = f"""
    SELECT
      mode,
      COUNT(*) AS total_trips,
      ROUND(AVG(distance_miles), 2) AS avg_distance_miles,
      ROUND(AVG(duration_minutes), 2) AS avg_duration_min,
      MIN(distance_miles) AS min_distance_miles,
      MAX(distance_miles) AS max_distance_miles,
      MIN(duration_minutes) AS min_duration_min,
      MAX(duration_minutes) AS max_duration_min
    FROM `{PROJECT_DATASET}.{table_name}`
    WHERE
      -- Oakley area (Contra Costa County)
      start_lat BETWEEN 37.98 AND 38.02
      AND start_lng BETWEEN -121.73 AND -121.67
      -- Palo Alto area (San Mateo County)
      AND end_lat BETWEEN 37.39 AND 37.47
      AND end_lng BETWEEN -122.18 AND -122.10
      AND distance_miles > 0 AND duration_minutes > 0
    GROUP BY mode
    ORDER BY total_trips DESC
    """
    return client.query(query).to_dataframe()

# Create one map centered between Oakley and Palo Alto
# Oakley: ~37.998, -121.712 | Palo Alto: ~37.442, -122.143
m = folium.Map(location=[37.72, -121.93], zoom_start=9, tiles="CartoDB positron")

# Helpful plugins
plugins.Fullscreen().add_to(m)
plugins.MiniMap(toggle_display=True).add_to(m)
plugins.MousePosition(position="bottomleft", prefix="Lat/Lng").add_to(m)

# Add region markers for reference
oakley_marker = folium.Marker(
    location=[37.998, -121.712],
    popup="Oakley, Contra Costa County",
    tooltip="Trip Origins",
    icon=folium.Icon(color='green', icon='play')
)
oakley_marker.add_to(m)

palo_alto_marker = folium.Marker(
    location=[37.442, -122.143], 
    popup="Palo Alto, San Mateo County",
    tooltip="Trip Destinations",
    icon=folium.Icon(color='red', icon='stop')
)
palo_alto_marker.add_to(m)

# Collect and display statistics for Oakley->Palo Alto trips
print("🗺️  OAKLEY TO PALO ALTO TRIP ANALYSIS")
print("="*60)

total_trip_stats = {}

for dataset_label, table_name in DATASETS:
    print(f"\n📊 Processing {dataset_label}...")
    
    # Get detailed statistics
    details = fetch_oakley_palo_alto_details(table_name)
    if not details.empty:
        total_trip_stats[dataset_label] = details
        total_trips = details['total_trips'].sum()
        print(f"   Total Oakley→Palo Alto trips: {total_trips}")
        
        if total_trips > 0:
            print("   Mode breakdown:")
            for _, row in details.iterrows():
                pct = (row['total_trips'] / total_trips) * 100
                print(f"     • {row['mode']}: {row['total_trips']} trips ({pct:.1f}%)")
                print(f"       Distance: {row['avg_distance_miles']:.1f} mi (range: {row['min_distance_miles']:.1f}-{row['max_distance_miles']:.1f})")
                print(f"       Duration: {row['avg_duration_min']:.1f} min (range: {row['min_duration_min']:.1f}-{row['max_duration_min']:.1f})")
    else:
        print("   No trips found for this dataset")

print("\n" + "="*60)

for dataset_label, table_name in DATASETS:
    print(f"\n🛣️  Fetching network trajectories for {dataset_label}...")
    df = fetch_flows_with_network_segments(table_name)
    if df.empty:
        print(f"   No trips with network segments found for {dataset_label}")
        continue

    # Parent layer per dataset (start with Q4 Thursday visible for demo)
    show_layer = (dataset_label == "Q4 Thursday")
    parent_group = folium.FeatureGroup(name=f"{dataset_label} (Network Routes)", show=show_layer)
    parent_group.add_to(m)

    # Color map per mode in this dataset
    modes = list(pd.Series(df["mode"].astype(str).fillna("UNKNOWN")).unique())
    color_cycle = cycle(PALETTE)
    color_by_mode = {mode: (next(color_cycle)) for mode in modes}

    print(f"   Found {len(df)} route patterns across {len(modes)} modes: {', '.join(modes)}")
    
    # Check for routes without network data
    routes_without_data = df[df['trajectory_segments'].isna() | (df['trajectory_segments'] == '')]
    if len(routes_without_data) > 0:
        print(f"   ⚠️  Found {len(routes_without_data)} route patterns without network trajectory data")
        for _, row in routes_without_data.iterrows():
            print(f"     - {row['mode']}: {row['trip_count']} trips missing network data")

    # Subgroup per mode
    for mode in modes:
        sub = df[df["mode"] == mode]
        sub_group = plugins.FeatureGroupSubGroup(parent_group, name=f"{dataset_label} — {mode}")
        sub_group.add_to(m)

        # Draw network trajectories instead of straight lines
        for _, r in sub.iterrows():
            # Enhanced tooltip for Oakley->Palo Alto trips showing route patterns
            tooltip_text = (f"Oakley→Palo Alto | {mode} | "
                          f"trips: {int(r.trip_count)} (same route) | "
                          f"~{r.avg_distance_miles:.0f}mi, {r.avg_duration_min:.0f}min")
            
            # Parse trajectory coordinates from network segments
            trajectory_coords = parse_trajectory_coordinates(r.trajectory_segments)
            
            if trajectory_coords and len(trajectory_coords) > 1:
                # Draw actual network trajectory
                folium.PolyLine(
                    locations=trajectory_coords,
                    color=color_by_mode[mode],
                    weight=2 + (float(r.trip_count) ** 0.40),
                    opacity=0.8,
                    tooltip=tooltip_text + " (Continuous Route)"
                ).add_to(sub_group)
            else:
                # Fallback to straight line if no network data
                folium.PolyLine(
                    locations=[(float(r.start_lat), float(r.start_lng)),
                             (float(r.end_lat), float(r.end_lng))],
                    color=color_by_mode[mode],
                    weight=2 + (float(r.trip_count) ** 0.40),
                    opacity=0.6,
                    tooltip=tooltip_text + " (Direct Line - No Network Data)",
                    dashArray="5, 10"  # Dashed line to indicate it's not actual route
                ).add_to(sub_group)

        # Optional: clustered start points for this mode
        cluster = plugins.MarkerCluster(name=f"{dataset_label} — {mode} (origins)")
        for _, r in sub.sample(min(len(sub), 500), random_state=0).iterrows():
            folium.CircleMarker(
                location=(float(r.start_lat), float(r.start_lng)),
                radius=3,
                fill=True, opacity=0.6, fill_opacity=0.6,
                color=color_by_mode[mode],
                popup=f"Origin: {mode} trips"
            ).add_to(cluster)
        cluster.add_to(sub_group)

# Grouped control makes the menu cleaner; regular LayerControl also works
try:
    plugins.GroupedLayerControl().add_to(m)
except Exception:
    folium.LayerControl(collapsed=False).add_to(m)

# Enhanced legend for network-based Oakley-Palo Alto analysis
legend_html = """
<div style="
 position: fixed; bottom: 10px; left: 10px; z-index: 9999;
 background: white; padding: 10px 12px; border: 1px solid #999; border-radius: 6px;
 font: 12px/1.2 Arial, sans-serif; max-width: 320px;">
  <b>🗺️ Oakley → Palo Alto Network Routes</b><br/>
  <small>Contra Costa County → San Mateo County</small><br/><br/>
  
  <b>🛣️ Route Types:</b><br/>
  • <strong>Solid lines</strong> = Continuous network trajectories<br/>
  • <strong>Dashed lines</strong> = Direct line (no network data)<br/>
  • Line thickness = Trip frequency<br/>
  • Colors = Transport modes<br/><br/>
  
  <b>📍 Markers:</b><br/>
  • 🟢 Green marker = Oakley (origin)<br/>
  • 🔴 Red marker = Palo Alto (destination)<br/>
  • Clustered dots = Trip origin points<br/><br/>
  
  <b>🎛️ Controls:</b><br/>
  • Toggle datasets (Q2/Q4, Thu/Sat)<br/>
  • Toggle transport modes<br/>
  • Hover routes for trip details<br/><br/>
  
  <b>💡 Network Analysis:</b><br/>
  • Shows sequential road segments taken<br/>
  • Connects segments at intersections<br/>
  • Reveals actual travel paths<br/>
  • Cross-county mobility insights<br/>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

output_filename = "oakley_palo_alto_world.html"
m.save(output_filename)
print(f"\n🎉 Saved network trajectory map: {output_filename}")
print("🛣️  Map shows continuous network routes using ordered road segments")
print("📊 Includes all trips (no 1000 limit) with network link data")
print("🔍 Solid lines = continuous trajectories, dashed = missing network data")
