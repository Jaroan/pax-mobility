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

for label, table_name in datasets:
    season_prefix = table_name[:table_name.rfind("_")]  # e.g., 'ca_2024_Q2_thursday'
    # Simplify to just prefix until the season, e.g., 'ca_2024_Q2'
    season_prefix = "_".join(season_prefix.split("_")[:3])  # ['ca', '2024', 'Q2'] => 'ca_2024_Q2'
    population_table = f"{season_prefix}_population"

    query = f"""
    SELECT
      ST_ASTEXT(ST_GEOGPOINT(start_lng,start_lat)) AS origin_wkt,
      trip.mode,
      COUNT(*) AS trip_count
    FROM `replica-customer.nasa_ames.{table_name}` AS trip
    JOIN `replica-customer.nasa_ames.{population_table}` AS pers
      ON pers.person_id = trip.person_id
    GROUP BY origin_wkt, trip.mode
    ORDER BY trip_count DESC
    LIMIT 500
    """

    df = client.query(query).to_dataframe()
    df['geometry'] = df['origin_wkt'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    m = folium.Map(location=[37.77, -122.42], zoom_start=10, title=label)
    for _, row in gdf.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=2 + row.trip_count**0.2,
            color='blue' if row.mode in ('PRIVATE_AUTO','CARPOOL','ON_DEMAND_AUTO') else 'green',
            fill=True,
            fill_opacity=0.6,
            popup=f"{label}\nMode: {row.mode}\nTrips: {row.trip_count}"
        ).add_to(m)
    fname = f"bay_area_trips_{label.replace(' ', '_')}.html"
    m.save(fname)
    print(f"Saved map: {fname}")
