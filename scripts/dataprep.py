import pandas as pd
from pathlib import Path

def load_data(city: str) -> pd.DataFrame:
    '''
    Takes a city string and returns a pandas DataFrame. "datetime_utc" is a 
    datetime64[ns, UTC] object.
    '''

    base = Path.home() / "buccs_notebooks" / "data" / "temperature" / city

    if city == "dortmund":
        meta = pd.read_csv(base / "stations_metadata.csv",
                           usecols=["station_id", "latitude", "longitude"])

        files = sorted((base / "hourly").glob("*.csv"))

        obs = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

        df = obs.merge(meta, on="station_id", how="left")

        df = df.rename(columns={"measured_at": "datetime_utc"})

        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    return df[["station_id", "latitude", "longitude", "datetime_utc", "air_temperature"]]