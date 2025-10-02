
## Open Meteo libraries
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

### Open - Meteo API Function ###

def fetch_weather_data(lat, lon, start_date, end_date):
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "surface_pressure", "direct_radiation"],
        "models": "best_match",
        "wind_speed_unit": "ms",
        "start_date": start_date,
        "end_date": end_date,
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    #print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    #print(f"Elevation: {response.Elevation()} m asl")
    #print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(3).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(4).ValuesAsNumpy()
    hourly_direct_radiation = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {"timestamp": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}

    hourly_data["temperature"] = hourly_temperature_2m
    hourly_data["humidity"] = hourly_relative_humidity_2m
    hourly_data["wind_speed"] = hourly_wind_speed_10m
    hourly_data["wind_direction"] = hourly_wind_direction_10m
    hourly_data["pressure"] = hourly_surface_pressure
    hourly_data["solar_radiation"] = hourly_direct_radiation

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    #hourly_dataframe["date"] = hourly_dataframe["date"] + pd.Timedelta(hours=3)
    #print("\nHourly data\n", hourly_dataframe)
    return hourly_dataframe


