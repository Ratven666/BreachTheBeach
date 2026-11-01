from matplotlib import pyplot as plt

from src.weather_history.domain import WeatherLayerWrapper

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

point = weather_wrapper.get_point("point_00000")

rose = point.wind_rose
print(rose.table_data.as_dataframe().head())

point.plot_wind_rose_matplotlib(
    output_path="output/point_00000_wind_rose.png",
    nsector=64,
)


point.plot_wind_rose_plotly(
    output_path="output/point_00000_wind_rose.html",
    nsector=64,
)