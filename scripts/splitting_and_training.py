from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



def split_data(X, y, mode="random"):
    if mode == "random":
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_val, y_train, y_val

    elif mode == "random_spatial":
        stations = X["station_id"].unique()

        rng = np.random.default_rng(42)
        n_test = int(round(0.3 * len(stations)))
        test_stations = rng.choice(stations, size=n_test, replace=False)

        # boolean mask over the ORIGINAL index — same rows selected in X and y
        test_mask = X["station_id"].isin(test_stations)

        X_train, X_val = X[~test_mask], X[test_mask]
        y_train, y_val = y[~test_mask], y[test_mask]

        X_train = X_train.drop(columns=["station_id"]).copy()
        X_val = X_val.drop(columns=["station_id"]).copy()

        return X_train, X_val, y_train, y_val

        
    elif mode == "block_spatial":
        lats = X["latitude"].unique()
        lons = X["longitude"].unique()

        min_lat, max_lat = lats.min(), lats.max()
        min_lon, max_lon = lons.min(), lons.max()

        cutoff_lat = np.mean([min_lat, max_lat])
        cutoff_lon = np.mean([min_lon, max_lon])

        ne_mask = (X["latitude"] >= cutoff_lat) & (X["longitude"] >= cutoff_lon)
        # se_mask = X["latitude"] < cutoff_lat and X["longitude"] >= cutoff_lon
        # nw_mask = X["latitude"] >= cutoff_lat and X["longitude"] < cutoff_lon
        # sw_mask = X["latitude"] < cutoff_lat and X["longitude"] < cutoff_lon

        X_train, X_val = X[~ne_mask], X[ne_mask]
        y_train, y_val = y[~ne_mask], y[ne_mask]

        X_train = X_train.drop(columns=["station_id"]).copy()
        X_val = X_val.drop(columns=["station_id"]).copy()

        return X_train, X_val, y_train, y_val