from src.weather_history.domain.WeatherLayerWrapper import WeatherLayerWrapper

weather_wrapper = WeatherLayerWrapper.from_file(
    "data/weather_daily_grid.geojson",
    working_crs="EPSG:32637",
)

assigned_gdf = weather_wrapper.assign_to_points(
    coastline_points_path="data/novoross_main_radius_points.geojson",
    strategy="idw",
    idw_k=4,
    idw_power=2.0,
    working_crs="EPSG:32637",
    output_geojson_path="output/coastline_points_with_weather.geojson",
)

for point in weather_wrapper:
    print(point.brief_dict())
    print(point.timeseries_df().head())

p0 = weather_wrapper.get_point("point_00000")
print(p0.timeseries_df().head())

weather_wrapper.export_all_points_weather(
    assigned_gdf=assigned_gdf,
    output_path="output/all_points_weather.geojson",
)

weather_wrapper.export_point_files(
    assigned_gdf=assigned_gdf,
    output_dir="output/point_weather_files",
)