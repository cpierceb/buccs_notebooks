import pandas as pd
from pathlib import Path
from thermofeel import calculate_relative_humidity_percent
import numpy as np
import xarray as xr
from pathlib import Path
from pyproj import Transformer
from tqdm import tqdm
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


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


#-----------------------------------------------------------------------------------------------------#
#-------------------------------------------- RASTERS--- ---------------------------------------------#
#-----------------------------------------------------------------------------------------------------#


def coarsen_raster(src_path, dst_path, target_res):
    with rasterio.open(src_path) as src:
        # obtain the current pixel size, e.g. (10.0, 10.0)
        cur_x, cur_y = src.res            
        print(f"current pixel size is {cur_x}, {cur_y}")

        # we only want to coarsen the rasters bc they are already given at the finest resolution possible
        if target_res < max(cur_x, cur_y):
            raise ValueError(
                f"target {target_res} m is finer than source "
                f"{cur_x}x{cur_y} m — this only coarsens"
            )

        band = src.read(1)

        # we detect any noData values
        msk = src.read_masks(1)

        # and set them to 0
        band[msk == 0] = 0

        # calculate the scaling factor
        fx = cur_x / target_res
        fy = cur_y / target_res

        # the new raster will contain less pixels in each direction, scaled by the factor
        new_w = max(1, round(src.width * fx))
        new_h = max(1, round(src.height * fy))

        # we initialize the empty coarsened raster with these pixel counts
        dst_arr = np.empty((new_h, new_w), dtype="float32")

        # we ensure that the new raster conserves its georeferencing: we multiply the fine transform(fine pixels x CRS)
        # by the scaling factor to obtain the new transform(coarse pixels x CRS)
        transform = src.transform * src.transform.scale(src.width / new_w,
                                                        src.height / new_h)
        reproject(
            band, dst_arr,                                # we reproject from source to destination arrays
            src_transform=src.transform, src_crs=src.crs, # from the source transform and CRS
            dst_transform=transform, dst_crs=src.crs,     # to the coarsened geospatial transform, but same source CRS
            resampling=Resampling.average,                # with the average resampling method
        )

        # we copy the metadata from the source raster
        profile = src.profile.copy()

        # we update the metadata with the new amount of pixels and transform
        profile.update(height=new_h, width=new_w, transform=transform)

    # we write to the destination file to save the new raster
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(dst_arr, 1)





# we prepare a helper function that can extract the geospatial data at different buffer radii
def sample_point_and_buffers(band, transform, res, x, y,
                             radii=(50, 100, 250, 500, 750, 1000)):
    """band: 2D array with nodata ALREADY filled to 0. Returns dict of predictors."""

    # transform.rowcol returns the raster cell that that x and y fall into
    row, col = rasterio.transform.rowcol(transform, x, y)
    out = {}

    # we check if the point measurement is actually contained in the extent of the raster
    if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
        out["nearest"] = band[row, col]
    else:
        print("out of bounds")
        out["nearest"] = np.nan

    if not radii:                 # nearest-only: skip all the window/buffer work
        return out
        
    # next we extract data from the maximum buffer radius in our list
    # rmax returns the maximum amount of cells in the horizontal or vertical direction
    rmax = int(np.ceil(max(radii) / res))
    # we compute which row and column this radius would absorb
    # clipping it to 0 (minmum raster extent) on the low end and the maximum extent of the raster on the high end
    r0, r1 = max(0, row - rmax), min(band.shape[0], row + rmax + 1)
    c0, c1 = max(0, col - rmax), min(band.shape[1], col + rmax + 1)

    # we extract this large square once, which will contain all the data needed for the circular buffers
    window = band[r0:r1, c0:c1]

    # ogrid is an efficient way to build a grid
    # wr is a column vector holding row indices and wc a row vector containing column indices
    wr, wc = np.ogrid[r0:r1, c0:c1]

    # dist computes the distances from each pixel in the window to the center pixel (i.e. the station)
    # the value is in meters after being multiplied by res
    dist = np.sqrt((wr - row) ** 2 + (wc - col) ** 2) * res

    # we iterate over the radii 
    for radius in radii:
        # we slice the maximum window to cells that are within the radius (dist contains every pixel's distance to the station)
        # this effectively creates a pixelated circle around the station
        vals = window[dist <= radius]
        # we return a column to the dataframe named after the buffer radius containing the mean of the values within the buffer
        out[f"buf_{radius}"] = np.mean(vals) if vals.size else np.nan

    return out



def build_station_predictors(df, raster_paths, radii=(50, 100, 250, 500, 750, 1000)):
    # we want to isolate individual stations
    stn = (df.groupby("station_id")                 # groupby groups our dataframe by station id
             .agg(latitude=("latitude", "first"),   # .agg reduces dimensionality: here we take the first lat and lon 
                  longitude=("longitude", "first")) # because all datetime entries for a single station have the same lat and lon
             .reset_index())                        # we reset the index so that station_id becomes a data column and not an index column
                                                    
    # we prepare a CRS transformer to convert lat/lon to EPSG:3035 northing/easting
    to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    stn["x"], stn["y"] = to_3035.transform(stn["longitude"].values,
                                           stn["latitude"].values)

    # We loop over each raster file first to open them only once
    for name, path in raster_paths.items():         # raster_paths is a dictionary containing a key (the name) and a value (the raster path)
        with rasterio.open(path) as src:
            band = src.read(1).astype("float32")
            msk = src.read_masks(1)                 
            band[msk == 0] = 0                      # we set noData values to 0 
            if name in ("tcd", "imp"):
                band[band == 255] = 0               # water tiles have a value of 255 in the Copernicus HRL                    
            transform, res = src.transform, src.res[0]

        # pandas iterrows iterates one row at a time, i.e. one station at a time
        for i, r in stn.iterrows():                 # i is the row index and r the row contents 

            # we now call our helper function, which takes the following arguments:
            s = sample_point_and_buffers(
                band,                               # band contains the data
                transform,                          # transform contains the pixel-geospatial transformation info,                        
                res,                                # which we need to go to/from pixel-geospatial coordinates referencing
                r["x"], r["y"],                     # this station's coordinates in the raster's CRS
                radii)                              # a range of distances in meters, optional as we have a hardcoded default 
            # s returns a dict of values (from nearest to buffer radius 1000)

            # assigns the raster value according to what pixel the station falls into, to a new column on our dataframe
            stn.loc[i, f"{name}_nearest"] = s["nearest"] 
            # we iteratively assign the buffer radius values to the dataframe with dynamic column naming
            for radius in radii:
                stn.loc[i, f"{name}_buf{radius}"] = s[f"buf_{radius}"] 

    return stn   # one row per station, with all geo predictor columns


#-----------------------------------------------------------------------------------------------------#
#-------------------------------------------------- ERA5 ---------------------------------------------#
#-----------------------------------------------------------------------------------------------------#


# our ERA5-Land data covers all of Europe and is stored daily, so we pass the days and location we need
def load_era5land(nc_dir, geopot_path, lat0, lon0, needed_dates):
    nc_dir = Path(nc_dir)
    vars_ = ["t2m", "d2m", "ssrd", "tp", "u10", "v10", "sp"]
    frames = []
    missing = []

    # we iterate over all daily files, with tqdm to give us progress information
    for d in tqdm(needed_dates, desc=f"ERA5-Land ({len(needed_dates)} days)"):
        f = nc_dir / f"ERA5Land_{d}.nc"       # d formatted yyyy-mm-dd
        if not f.exists():
            missing.append(f.name)
            continue
        with xr.open_dataset(f) as ds:
            # we select the daily point data, .squeeze() removes the geographical coordinates as dimensions
            # so we end up with a single time-series of data
            pt = ds[vars_].sel(latitude=lat0, longitude=lon0, method="nearest").squeeze()

            # we convert the Dataset to a pandas DataFrame, reset_index makes our single column a data column
            # and not an index column
            frames.append(pt.to_dataframe().reset_index())

    if missing:
        print(f"warning: {len(missing)} needed files absent, e.g. {missing[:3]}")
    if not frames:
        raise FileNotFoundError(f"no matching ERA5-Land files found in {nc_dir}")

    # we concatenate our daily dataframes into one, we renumber our index 0 to N
    met = pd.concat(frames, ignore_index=True)

    # we rename our time column to ensure our dataframes match up later
    met = met.rename(columns={"valid_time": "datetime_utc"})

    # we ensure our datetime is tz-aware
    met["datetime_utc"] = pd.to_datetime(met["datetime_utc"], utc=True)

    
    met = (met[["datetime_utc", *vars_]]
           .drop_duplicates("datetime_utc").sort_values("datetime_utc")
           .reset_index(drop=True))

    gz = xr.open_dataset(geopot_path).sel(latitude=lat0, longitude=lon0,
                                          method="nearest").squeeze()
    era5_elev = float(gz["z"].values) / 9.80665
    gz.close()
    return met, era5_elev


def deaccumulate(met, col):
    out = met.sort_values("datetime_utc").copy()    # make sure the date_time utc is sorted
    hourly = out[col].diff()                        # this is the difference between subsequent timesteps
    reset = out["datetime_utc"].dt.hour == 1        # first step of daily window, taken as-is
    hourly[reset] = out.loc[reset, col]             # reset the values at 01:00 UTC

    # detect gaps: time since previous row should be exactly 1 hour
    dt = out["datetime_utc"].diff()
    
    gap = dt > pd.Timedelta(hours=1)
    # at a gap (non-01:00), the diff is untrustworthy -> NaN it
    hourly[gap & ~reset] = np.nan

    hourly = hourly.clip(lower=0)                   # make sure all values are positive
    out[f"{col}_deac"] = hourly                     # create a new column with suffic _deac
    return out



def transform_met(met):
    m = met.copy()
    m["rh"]  = calculate_relative_humidity_percent(m["t2m"], m["d2m"]) # rh calculation from ecmwf, needs values in K
    # now we convert our temperature columns to °C
    m["t2m"] = m["t2m"] - 273.15
    m["d2m"] = m["d2m"] - 273.15
    # and calculate scalar windspeed
    m["wspd"] = np.sqrt(m["u10"]**2 + m["v10"]**2)
    return m