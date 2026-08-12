from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xgboost as xgb



def split_data(X, y, mode = "random", quadrant = "ne"):
    
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

        masks = {
            "ne" : (X["latitude"] >= cutoff_lat) & (X["longitude"] >= cutoff_lon),
            "se" : (X["latitude"] < cutoff_lat) & (X["longitude"] >= cutoff_lon),
            "nw" : (X["latitude"] >= cutoff_lat) & (X["longitude"] < cutoff_lon),
            "sw" : (X["latitude"] < cutoff_lat) & (X["longitude"] < cutoff_lon)
        }

        val_mask = masks[quadrant] 

        X_train, X_val = X[~val_mask], X[val_mask]
        y_train, y_val = y[~val_mask], y[val_mask]

        X_train = X_train.drop(columns=["station_id"]).copy()
        X_val = X_val.drop(columns=["station_id"]).copy()

        return X_train, X_val, y_train, y_val


def train_model(X_train, X_val, y_train, y_val, model_type):
    if model_type == "xgboost":
        # params = {

        #     "eta": 0.3,
        #     "max_depth": 10,
        #     "colsample_bytree": 0.8,
        #     "subsample": 0.8,
        #     "gamma": 1.0,
        #     "min_child_weight": 1.0,
        #     "reg_lambda": 1.0,
        #     "reg_alpha": 0.0
        # }

        features = list(X_train.columns)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

        dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

        watchlist = [(dtrain, "train"), (dval, "valid")]

        # params["objective"] = "reg:squarederror"
        # params["eval_metric"] = "rmse" 

        params = {"objective": "reg:squarederror", "eval_metric": "rmse"}

        model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, 
                            early_stopping_rounds=10, verbose_eval=15)


        y_pred = model.predict(dval)

        return y_pred, model