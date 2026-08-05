import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from pyproj import Transformer
import cmcrameri.cm as cmc



def obs_map(data: pd.DataFrame):
    # we take the mean observed temperature for each station and in so doing collapse the datafame
    # to one row per station
    stn = (data.groupby("station_id")
                .agg(lon=("longitude", "first"),
                     lat=("latitude", "first"),
                     tmean=("air_temperature", "mean"))
                .reset_index())
    
    # we want to plot using a projection that is suited to all of Europe instead of Mercator
    # x and y represent easting and northing in this CRS, respectively
    to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    stn["x"], stn["y"] = to_3035.transform(stn["lon"].values, stn["lat"].values)


    # we set up the extent of the figure by adding a bit of padding around the extreme locations 
    # of the stations
    padx = (stn["x"].max() - stn["x"].min()) * 0.05
    pady = (stn["y"].max() - stn["y"].min()) * 0.05

    # we create the figure with the given dimensions
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(stn["x"].min() - padx, stn["x"].max() + padx)
    ax.set_ylim(stn["y"].min() - pady, stn["y"].max() + pady)

    # scatter the stations on the map and color the dots according to their mean temperature
    sc = ax.scatter(stn["x"], stn["y"], c=stn["tmean"],
                    cmap=cmc.roma_r, s=45, edgecolors="none")

    # # small station-id labels, nudged off each dot
    # for _, r in stn.iterrows():
    #     ax.annotate(r["station_id"], (r["x"], r["y"]),
    #                 xytext=(3, 3), textcoords="offset points",
    #                 fontsize=6, color="0.2")

    # add a basemap to situate ourselves
    ctx.add_basemap(ax, crs="EPSG:3035",
                    source=ctx.providers.CartoDB.Positron)

    ax.set_xticks([]); ax.set_yticks([])

    # colorbar = legend
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Mean annual air temperature (°C)")

    plt.show()

    return fig, ax



    