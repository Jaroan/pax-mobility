from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

query = """
SELECT
  origin_lat,
  origin_lon,
  dest_lat,
  dest_lon,
  passengers,
  day_type,
  quarter
FROM
  `your_project.your_dataset.trips_table`
WHERE
  quarter IN ('Q2', 'Q4')
  AND day_type = 'weekday'
"""
df = client.query(query).to_dataframe()

import geopandas as gpd
import folium

# parse WKT into GeoSeries
gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["origin_wkt"]))
# Create a base map centered on Bay Area
m = folium.Map(location=[37.77, -122.42], zoom_start=10)
# Add trips as circle markers
for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=row.trip_count ** 0.3,
        color='blue' if row.mode == 'PRIVATE_AUTO' else 'red',
        fill=True,
        fill_opacity=0.5
    ).add_to(m)
m.save("trip_map.html")
