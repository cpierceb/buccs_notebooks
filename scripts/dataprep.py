import pandas as pd
from pathlib import Path


def load_data(city: str) -> pd.DataFrame:
    '''
    Takes a city string and returns a pandas DataFrame. "datetime_utc" is a 
    datetime64[ns, UTC] object.
    '''
    base = Path.home() / "buccs_notebooks" / "data" / "temperature" / city
    pkl = base / f"{city}_combined.pkl"

    if pkl.exists():
        print("loading pre-existing dataframe")
        return pd.read_pickle(pkl)

    else:
        print("reading raw data")
        if city == "dortmund":
            # this groups all of the csv observation files together
            files = sorted((base / "hourly").glob("*.csv"))

            # we concatenate all files together
            obs = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

            # the metadata contains a lot of columns, so let's just take the most important ones
            meta = pd.read_csv(base / "stations_metadata.csv",
                            usecols=["station_id", "latitude", "longitude"])

            # we merge the metadata to the observations
            df = obs.merge(meta, on="station_id", how="left")

            df = df.rename(columns={"measured_at": "datetime_utc"})

            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

        else:  # bern, freiburg, ghent — SEF/TSV, one file per station
            files = sorted(base.glob("*.tsv"))

            records = []
            #there's extra processing involved here, so we open files one by one
            for path in files:
                raw, meta = read_sef(path)

                station_id = meta.get("ID")

                # we only keep hourly measuremets, i.e. when minute = 0
                raw = raw[raw["Minute"] == 0].copy()         

                # we force the observed value to numeric       
                raw["Value"] = pd.to_numeric(raw["Value"], errors="coerce")

                # so that we can drop any non numeric or NaN values
                raw = raw.dropna(subset=["Value"])

                # drop QC-flagged rows; keep clean + benign 'orig.time' annotations
                raw = raw[
                    raw["Meta"].isna()
                    | (raw["Meta"].str.strip() == "")
                    | raw["Meta"].str.strip().str.startswith("orig.time", na=False)
                ]

                # we convert our separate date time columns to the same format as the Dortmund observations
                raw["datetime_utc"] = pd.to_datetime(
                    raw[["Year", "Month", "Day", "Hour", "Minute"]].assign(Second=0),
                    utc=True,
                )

                # we rename our columns to the standard
                raw["station_id"] = station_id
                raw["latitude"] = float(meta.get("Lat", "nan"))
                raw["longitude"] = float(meta.get("Lon", "nan"))
                raw = raw.rename(columns={"Value": "air_temperature"})

                # we append the dataframe with every iteration for every station
                records.append(raw[["station_id", "latitude", "longitude",
                                    "datetime_utc", "air_temperature"]])


            df = pd.concat(records, ignore_index=True)

        df = df[["station_id", "latitude", "longitude", "datetime_utc", "air_temperature"]]
        df = df.dropna(subset=["air_temperature"])
        df.to_pickle(pkl)

    return df


#-----------------------------------------------------------------------------------------------------#
#-------------------------------------- HELPER FUNCTIONS ---------------------------------------------#
#-----------------------------------------------------------------------------------------------------#
def read_sef(path):
    """One SEF/TSV file -> (data DataFrame, metadata dict). Data starts at the 'Year' header line.
    This is a helper function for the data loading function, as we want to parse both the metadata
    and the actual data from the .tsv files"""
    metadata = {}

    # this part is very specific to the header, so before the data starts
    header_line = None
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t", 1)
            if parts and parts[0] == "Year":
                header_line = i
                break
            if len(parts) == 2:
                metadata[parts[0]] = parts[1]
    if header_line is None:
        raise ValueError(f"no 'Year' data header in {path}")

    # and finally we can read the actual data with what comes after the metadata
    df = pd.read_csv(
        path, sep="\t", header=header_line,
        dtype={"Year": "Int64", "Month": "Int64", "Day": "Int64",
               "Hour": "Int64", "Minute": "Int64", "Period": str, "Meta": str},
        na_values=["NA", "   NA"], keep_default_na=True, low_memory=False,
    )
    return df, metadata