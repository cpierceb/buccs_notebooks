from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------------------------------------------#
# ---------------------------------------- SPLITTING ------------------------------------------#
# ---------------------------------------------------------------------------------------------#

def split_data(X, y, mode = "random", quadrant = "sw"):
    
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


# ---------------------------------------------------------------------------------------------#
# ------------------------------------- FEATURE SELECTION -------------------------------------#
# ---------------------------------------------------------------------------------------------#

def feature_selection(model_type, X_tr, y_tr=None, X_val=None, y_val=None):
    if model_type == "xgboost":
        drop_cols = ["air_temperature", "temp_diff", "lcz_nearest",
                     "datetime_utc", "station_id",
                     "u10", "v10", "ssrd", "tp", "d2m", "t2m"]
        keep = [c for c in X_tr.columns if c not in drop_cols]
        print(keep)
        return keep                                    # list of feature names

    elif model_type == "lur":
        sel, r2 = select_forward_free(X_tr, y_tr, X_val, y_val)
        print(f"LUR: {len(sel)} predictors, recon_val_R²={r2:.4f}")
        return sel                                     # list of feature names


def val_r2_recon(X_tr, y_tr, X_val, y_val, cols):
    """Fit on train (residual target), predict val, reconstruct absolute temp, R² on that."""
    if not cols:
        return -np.inf
    model = sm.OLS(y_tr, sm.add_constant(X_tr[cols])).fit()
    pred_resid = model.predict(sm.add_constant(X_val[cols], has_constant="add"))
    obs_abs  = y_val.values + X_val["t2m_corr"].values
    pred_abs = pred_resid.values + X_val["t2m_corr"].values
    return r2_score(obs_abs, pred_abs)


def select_forward_free(X_tr, y_tr, X_val, y_val, threshold=0.0000):
    """Unrestricted forward selection: add the single best predictor each round."""
    exclude = {"station_id", "datetime_utc", "air_temperature", "temp_diff", "lcz_nearest"}
    remaining = [c for c in X_tr.columns
                 if c not in exclude and not c.startswith("LCZ_")]
    selected = []

    score = lambda cols: val_r2_recon(X_tr, y_tr, X_val, y_val, cols)
    current = score(selected)

    while remaining:
        best_pred, best_s = None, current
        for cand in remaining:
            s = score(selected + [cand])
            if s > best_s:
                best_pred, best_s = cand, s
        if best_pred is None or (best_s - current) < threshold:
            break
        selected.append(best_pred)
        remaining.remove(best_pred)
        current = best_s
        print(f"added {best_pred:20s} recon_val_R²={best_s:.4f}")

    return selected, current



# ---------------------------------------------------------------------------------------------#
# ----------------------------------------- TRAINING ------------------------------------------#
# ---------------------------------------------------------------------------------------------#

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
    

    elif model_type == "lur":
        X_tr_c  = sm.add_constant(X_train)
        X_val_c = sm.add_constant(X_val, has_constant="add")

        model = sm.OLS(y_train, X_tr_c).fit()
        y_pred = model.predict(X_val_c)

        print(model.params.sort_values(key=abs, ascending=False))
        
        return y_pred, model