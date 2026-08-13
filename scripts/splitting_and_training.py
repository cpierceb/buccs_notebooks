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
        # random train_test_split from sklearn
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        return X_train, X_val, y_train, y_val

    # slightly less random, we filter out an integer no. of stations
    elif mode == "random_spatial":
        stations = X["station_id"].unique()

        # set a random see for reproducibility
        rng = np.random.default_rng(42)

        # integer no. of stations
        n_test = int(round(0.3 * len(stations)))

        #randomly select the validation stations
        val_stations = rng.choice(stations, size=n_test, replace=False)

        # boolean mask for all validation stations
        val_mask = X["station_id"].isin(val_stations)

        # filter out the validation stations = training set
        # leave the validation stations in = validation set
        X_train, X_val = X[~val_mask], X[val_mask]
        y_train, y_val = y[~val_mask], y[val_mask]

        # make sure we drop station_id as we don't want strings in our predictors
        X_train = X_train.drop(columns=["station_id"]).copy()
        X_val = X_val.drop(columns=["station_id"]).copy()

        return X_train, X_val, y_train, y_val

    # even less random: we block out a whole quadrant of the city for validation
    elif mode == "block_spatial":
        lats = X["latitude"].unique()
        lons = X["longitude"].unique()

        #we determine the geographical extent of the station locations
        min_lat, max_lat = lats.min(), lats.max()
        min_lon, max_lon = lons.min(), lons.max()

        # we get the center points
        cutoff_lat = np.mean([min_lat, max_lat])
        cutoff_lon = np.mean([min_lon, max_lon])

        # and create boolean masks for the 4 quadrants
        masks = {
            "ne" : (X["latitude"] >= cutoff_lat) & (X["longitude"] >= cutoff_lon),
            "se" : (X["latitude"] < cutoff_lat) & (X["longitude"] >= cutoff_lon),
            "nw" : (X["latitude"] >= cutoff_lat) & (X["longitude"] < cutoff_lon),
            "sw" : (X["latitude"] < cutoff_lat) & (X["longitude"] < cutoff_lon)
        }

        # the quadrant is "sw" as a default, so all stations in the sw quadrant will be left out of training
        # boolean mask for all validation stations
        val_mask = masks[quadrant] 

        # filter out the validation stations = training set
        # leave the validation stations in = validation set
        X_train, X_val = X[~val_mask], X[val_mask]
        y_train, y_val = y[~val_mask], y[val_mask]

        X_train = X_train.drop(columns=["station_id"]).copy()
        X_val = X_val.drop(columns=["station_id"]).copy()

        return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------------------------#
# ------------------------------------- FEATURE SELECTION -------------------------------------#
# ---------------------------------------------------------------------------------------------#

def feature_selection(model_type, X_tr, y_tr, X_val, y_val, target):
    if model_type == "xgboost":
        # we manually drop the targets, lcz because it's categorical (we leave the one-hot encoded LCZs)
        # datetime_utc is a datetime object, station_id is a string
        # we take out the base ERA5 variables and only keep the transformed ones
        #(scalar windspeed, deaccumulated variables, corrected t2m, RH instead of d2m)
        drop_cols = ["air_temperature", "temp_diff", "lcz_nearest",
                     "datetime_utc", "station_id",
                     "u10", "v10", "ssrd", "tp", "d2m", "t2m"]
        keep = [c for c in X_tr.columns if c not in drop_cols]
        print(keep)
        # whatever columns are left = our features = predictor variables
        return keep                                    # list of feature names

    # LUR is more sensitive to the predictors, their colinearities etc., hence it's a bit more complex...
    elif model_type == "lur":
        sel, r2 = select_forward_free(X_tr, y_tr, X_val, y_val, target)
        print(f"LUR: {len(sel)} predictors, recon_val_R²={r2:.4f}")
        return sel                                     # list of feature names


def val_r2_recon(X_tr, y_tr, X_val, y_val, target, cols):
    """Fit on train (residual target), predict val, reconstruct absolute temp, R² on that."""
    if not cols:
        return -np.inf
    
    # fit an Ordinary Least Squares linear regression with response var = y_tr and 
    # explanatory vars = X_tr[cols], cols being passed through the function
    # we need to add a column of 1s (add_constant) => that is the intercept for the regression
    model = sm.OLS(y_tr, sm.add_constant(X_tr[cols])).fit()

    # we predict on Xval using the fitted model
    pred_resid = model.predict(sm.add_constant(X_val[cols], has_constant="add"))

    add_back = X_val["t2m_corr"].values if target == "temp_diff" else 0

    # the observed values are y_val (+ t2m_corr if we are using the residuals as our target)
    obs_abs  = y_val.values + add_back

    # the predicted values are from pred_resid (+ t2m_corr if we are using the residuals as our target)
    pred_abs = pred_resid.values + add_back

    return r2_score(obs_abs, pred_abs)


def select_forward_free(X_tr, y_tr, X_val, y_val, target, threshold=0.0000):
    """Unrestricted forward selection: add the single best predictor each round."""

    # we exclude columns that can't act as predictor variables, i.e. the targets, LCZs (categories) etc..
    exclude = {"station_id", "datetime_utc", "air_temperature", "temp_diff", "lcz_nearest", "t2m"}
    remaining = [c for c in X_tr.columns
                 if c not in exclude and not c.startswith("LCZ_")]

    # we initialize the selected predictors as an empty list
    selected = []

    score = lambda cols: val_r2_recon(X_tr, y_tr, X_val, y_val, target, cols)
    current = score(selected)

    while remaining:
        best_pred, best_s = None, current
        # we run individual models with 1 additional column each
        # the first run runs a linear regression for every predictor,
        for cand in remaining:
            # score returns the R2 from individual models
            s = score(selected + [cand])
            if s > best_s:
                # we keep the predictor that yielded the model with the highest R2
                best_pred, best_s = cand, s
        # we keep iterating, adding predictors one at a time until R2 stops improving
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

        # XGBoost requires DMatrix objects (XGBoost specific)
        # passing it like this allows us to keep training and validation sets consistent
        # and we also conserve the feature names 
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

        dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

        watchlist = [(dtrain, "train"), (dval, "valid")]

        # params["objective"] = "reg:squarederror"
        # params["eval_metric"] = "rmse" 

        params = {"objective": "reg:squarederror", "eval_metric": "rmse"}

        # evals knows that it should use dval for early stopping because we are training on dtrain
        model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, 
                            early_stopping_rounds=10, verbose_eval=15)


        y_pred = model.predict(dval)

        return y_pred, model
    

    elif model_type == "lur":
        # we add the intercept column
        X_tr_c  = sm.add_constant(X_train)
        X_val_c = sm.add_constant(X_val, has_constant="add")

        model = sm.OLS(y_train, X_tr_c).fit()
        y_pred = model.predict(X_val_c)

        # for the regression, we want to know the coefficients
        print(model.params.sort_values(key=abs, ascending=False))
        
        return y_pred, model