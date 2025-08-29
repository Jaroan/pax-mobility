from google.cloud import bigquery
import geopandas as gpd
import folium
from shapely import wkt

client = bigquery.Client()

# Define your datasets to process
datasets = [
    ("Q2 Thursday", "ca_2024_Q2_thursday_trip"),
    ("Q2 Saturday", "ca_2024_Q2_saturday_trip"),
    ("Q4 Thursday", "ca_2024_Q4_thursday_trip"),
    ("Q4 Saturday", "ca_2024_Q4_saturday_trip"),
]

# Initialize the map
m = folium.Map(location=[37.77, -122.42], zoom_start=10)

# Define transportation modes for classification
mode_colors = {
    'BIKING': 'pink',
    'PRIVATE_AUTO': 'blue',
    'CARPOOL': 'green',
    'ON_DEMAND_AUTO': 'purple',
    'WALKING': 'orange',
    'COMMERCIAL': 'red',
    'PUBLIC_TRANSIT': 'yellow',
    'OTHER_TRAVEL_MODE': 'grey'
}

# Create a FeatureGroup for each transportation mode
mode_layers = {mode: folium.FeatureGroup(name=mode) for mode in mode_colors}

for label, table_name in datasets:
    season_prefix = table_name[:table_name.rfind("_")]
    season_prefix = "_".join(season_prefix.split("_")[:3])
    population_table = f"{season_prefix}_population"

    query = f"""
    SELECT
      start_lat,
      start_lng,
      end_lat,
      end_lng,
      trip.mode,
      COUNT(*) AS trip_count
    FROM `replica-customer.nasa_ames.{table_name}` AS trip
    JOIN `replica-customer.nasa_ames.{population_table}` AS pers
      ON pers.person_id = trip.person_id
    GROUP BY start_lat, start_lng, end_lat, end_lng, trip.mode
    ORDER BY trip_count DESC
    LIMIT 500
    """

    df = client.query(query).to_dataframe()

    for _, row in df.iterrows():
        # Determine the color based on transportation mode
        color = mode_colors.get(row['mode'], 'gray')

        # Create a line from start to end coordinates
        line = folium.PolyLine(
            locations=[(row['start_lat'], row['start_lng']), (row['end_lat'], row['end_lng'])],
            color=color,
            weight=2,
            opacity=0.6
        )

        # Add the line to the corresponding mode layer
        mode_layers[row['mode']].add_child(line)

# Add all mode layers to the map
for layer in mode_layers.values():
    layer.add_to(m)

# Add LayerControl to toggle layers
folium.LayerControl().add_to(m)

# Save the map to an HTML file
m.save("interactive_trip_map.html")
print("Interactive map saved as 'interactive_trip_map.html'")