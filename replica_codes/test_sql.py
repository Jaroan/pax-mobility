SELECT
  origin_census_tract,
  destination_census_tract,
  COUNT(*) AS trip_count
FROM
  `your_project.your_dataset.replica_trip_table_Q2_weekday`
GROUP BY
  origin_census_tract, destination_census_tract;
