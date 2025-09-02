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

# Create one map and add dataset groups, then mode subgroups
m = folium.Map(location=[37.77, -122.42], zoom_start=10, tiles="CartoDB positron")

# Helpful plugins
plugins.Fullscreen().add_to(m)
plugins.MiniMap(toggle_display=True).add_to(m)
plugins.MousePosition(position="bottomleft", prefix="Lat/Lng").add_to(m)

for dataset_label, table_name in DATASETS:
    df = fetch_flows(table_name)
    if df.empty:
        continue

    # Parent layer per dataset (start hidden to avoid heavy initial render)
    parent_group = folium.FeatureGroup(name=f"{dataset_label}", show=False)
    parent_group.add_to(m)

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
            folium.PolyLine(
                locations=[(float(r.start_lat), float(r.start_lng)),
                           (float(r.end_lat),   float(r.end_lng))],
                color=color_by_mode[mode],
                weight=1 + (float(r.trip_count) ** 0.30),  # gentle scaling
                opacity=0.6,
                tooltip=f"{dataset_label} | {mode} | trips: {int(r.trip_count)}"
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

# Simple legend (colors are dataset-local, but gives quick hints)
legend_html = """
<div style="
 position: fixed; bottom: 10px; left: 10px; z-index: 9999;
 background: white; padding: 10px 12px; border: 1px solid #999; border-radius: 6px;
 font: 12px/1.2 Arial, sans-serif;">
  <b>How to use</b><br/>
  • Toggle datasets (Q2/Q4 Thu/Sat).<br/>
  • Within each dataset, toggle individual modes.<br/>
  • Line width ~ trip count per OD pair.<br/>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save("replica_bayarea_interactive_flows.html")
print("Saved: replica_bayarea_interactive_flows.html")
