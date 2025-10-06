import pandas as pd
import numpy as np

def feature_engineering(data: pd.DataFrame, future_data: pd.DataFrame, resample_freq: str = None, drop_columns = None) -> pd.DataFrame:
    if data is None:
        raise ValueError("Dataframe is None")
    
    if drop_columns is not None:
        data = data.drop(columns=drop_columns, errors='ignore')
        print(f"Dropped columns: {drop_columns}")

    data = data.copy()
    
    # Convert timestamp to datetime and adjust timezone if needed
    data['timestamp'] = pd.to_datetime(data['timestamp']) + pd.Timedelta(hours=3)
    future_data['timestamp'] = pd.to_datetime(future_data['timestamp']) + pd.Timedelta(hours=3)
    
    # Sort by timestamp
    data = data.sort_values("timestamp").reset_index(drop=True)
    future_data = future_data.sort_values("timestamp").reset_index(drop=True)

    # ===== Wind components =====
    data["wd_rad"] = np.deg2rad(data["wind_direction"])
    data["wind_u"] = np.cos(data["wd_rad"]) * data["wind_speed"]
    data["wind_v"] = np.sin(data["wd_rad"]) * data["wind_speed"]

    future_data["wd_rad"] = np.deg2rad(future_data["wind_direction"])
    future_data["wind_u"] = np.cos(future_data["wd_rad"]) * future_data["wind_speed"]
    future_data["wind_v"] = np.sin(future_data["wd_rad"]) * future_data["wind_speed"]


    # Resample if requested
    if resample_freq is not None:
        data = data.set_index('timestamp').resample(resample_freq).mean().reset_index()

    # Append future data
    data["status"] = "observed"
    future_data["status"] = "api"
    data = pd.concat([data, future_data[future_data["timestamp"] > data.timestamp.max()]], axis=0).reset_index(drop=True)


    # ===== Delta wind degree (geçmişe dayalı) =====
    for l in [1, 3, 6]:
        data[f"delta_wind_direction_{l}h"] = data["wind_direction"].diff(l)

    
    # ===== Cyclic time features =====
    data['hour'] = data['timestamp'].dt.hour
    data['day'] = data['timestamp'].dt.day
    data['month'] = data['timestamp'].dt.month

    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    return data

# ========================
# Meteorological Features
# ========================
def add_meteo_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # Dew point & Dew point depression
    data["dew_point"] = data["temperature"] - (100 - data["humidity"]) / 5.0
    data["dewpoint_dep"] = data["temperature"] - data["dew_point"]

    return data

# ========================
# Advanced Features
# ========================
def add_advanced_features_with_lags(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = data.sort_values("timestamp")

    # ===== Rolling mean & std =====
    rolling_cols = ["temperature", "humidity", "pressure"]
    windows = [3, 6, 12]  # saat
    for col in rolling_cols:
        for w in windows:
            data[f"{col}_rollmean_{w}h"] = data[col].rolling(window=w, min_periods=1).mean()
            data[f"{col}_rollstd_{w}h"] = data[col].rolling(window=w, min_periods=1).std().fillna(0)


    # ===== Interaction terms =====
    data["temp_dewpoint_interaction"] = data["temperature"] * data["dewpoint_dep"]
    data["wind_solar_interaction"] = data["wind_speed"] * data["solar_radiation"]

    return data


def generate_features(df: pd.DataFrame, rolling_windows=[3,6], lags=[1,2,3,6,12,24]) -> pd.DataFrame:
    """
    Feature engineering pipeline:
    - Wind components
    - Time cyclic features
    - Delta features (farklar)
    - Lag features (geçmiş değerler)
    - Rolling mean/std
    - Interaction terms
    - Wind direction categories
    """
    data = df.copy()
    data = data.sort_values("timestamp").reset_index(drop=True)


    # ===== Interaction terms =====
    data["temp_dewpoint_interaction"] = data["temperature"] * data["dewpoint_dep"]
    data["wind_solar_interaction"] = data["wind_speed"] * data["solar_radiation"]
    
    # ===== Delta features (differences) =====
    delta_cols = ["temperature", "humidity", "pressure", "wind_speed", "solar_radiation","temp_dewpoint_interaction","wind_solar_interaction","dew_point","dewpoint_dep"]
    for col in delta_cols:
        for l in lags:
            data[f"delta_{col}_{l}h"] = data[col] - data[col].shift(l)

    # ===== Lag features (shifted past values) =====
    for col in delta_cols:
        for l in lags:
            data[f"lag_{col}_{l}h"] = data[col].shift(l)

    # ===== Rolling mean & std =====
    rolling_cols = ["temperature", "humidity", "pressure"]
    for col in rolling_cols:
        for w in rolling_windows:
            data[f"{col}_rollmean_{w}h"] = data[col].rolling(window=w, min_periods=1).mean()
            data[f"{col}_rollstd_{w}h"] = data[col].rolling(window=w, min_periods=1).std().fillna(0)


    # ===== Wind direction categories (8 bins) =====
    #bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    #labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    #data["wind_dir_cat"] = pd.cut(data["wind_direction"] % 360, bins=bins, labels=labels, right=False)

    # ===== Delta wind direction =====
    #for l in [1, 3, 6]:
    #    data[f"delta_wind_direction_{l}h"] = data["wind_direction"].diff(l)

    return data
