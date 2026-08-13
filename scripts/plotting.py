import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap, Normalize
import contextily as ctx
import numpy as np
from pyproj import Transformer
import cmcrameri.cm as cmc
import rasterio



def obs_map(data: pd.DataFrame, value_col: str):
    stn = data.copy()
    # we want to plot using a projection that is suited to all of Europe instead of Mercator
    # x and y represent easting and northing in this CRS, respectively
    to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    stn["x"], stn["y"] = to_3035.transform(stn["longitude"].values, stn["latitude"].values)


    # we set up the extent of the figure by adding a bit of padding around the extreme locations 
    # of the stations
    padx = (stn["x"].max() - stn["x"].min()) * 0.05
    pady = (stn["y"].max() - stn["y"].min()) * 0.05

    # we create the figure with the given dimensions
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(stn["x"].min() - padx, stn["x"].max() + padx)
    ax.set_ylim(stn["y"].min() - pady, stn["y"].max() + pady)
    ax.set_aspect("equal") 

    if value_col == "tmean":
        # scatter the stations on the map and color the dots according to their mean temperature
        sc = ax.scatter(stn["x"], stn["y"], c=stn[value_col],
                        cmap=cmc.roma_r, s=45, edgecolors="none")
        # colorbar = legend
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Mean annual air temperature (°C)")

    elif value_col == "uhi_mean":
        # scatter the stations on the map and color the dots according to their mean temperature
        sc = ax.scatter(stn["x"], stn["y"], c=stn[value_col],
                        cmap=cmc.vik, s=45, edgecolors="none")
        # colorbar = legend
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Mean UHI (°C)")

    elif value_col == "rmse":
        # shared color scale so square (train) and circle (val) colors are comparable
        vmin, vmax = stn["rmse"].min(), stn["rmse"].max()

        train = stn[stn["split"] == "train"]
        val   = stn[stn["split"] == "val"]

        sc = ax.scatter(train["x"], train["y"], c=train["rmse"], cmap=cmc.batlow,
                        vmin=vmin, vmax=vmax, marker="s", s=55,
                        edgecolors="black", linewidths=0.4, label="train")
        ax.scatter(val["x"], val["y"], c=val["rmse"], cmap=cmc.batlow,
                   vmin=vmin, vmax=vmax, marker="o", s=55,
                   edgecolors="black", linewidths=0.4, label="val")

        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Station RMSE (°C)")
        ax.legend(loc="upper right")

    # add a basemap to situate ourselves
    ctx.add_basemap(ax, crs="EPSG:3035",
                    source=ctx.providers.CartoDB.Positron)

    ax.set_xticks([]); ax.set_yticks([])

    plt.show()

    return fig, ax



def week_plot(hot_week, cold_week, hot_id, cold_id):
    c_hot = cmc.vik(0.8)   # warm end
    c_cold = cmc.vik(0.2)  # cool end

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hot_week["datetime_utc"], hot_week["air_temperature"],
            color=c_hot, label=f"{hot_id} (warmest mean)")
    ax.plot(cold_week["datetime_utc"], cold_week["air_temperature"],
            color=c_cold, label=f"{cold_id} (coolest mean)")

    ax.yaxis.set_major_locator(MultipleLocator(5))   # major ticks every 5 °C
    ax.yaxis.set_minor_locator(MultipleLocator(1))   # minor ticks every 1 °C
    ax.grid(which="major", axis="y", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", axis="y", linewidth=0.4, alpha=0.3)
    ax.set_ylabel("Air temperature (°C)")
    ax.legend()
    fig.autofmt_xdate()      
    plt.show()
    return fig, ax


def quick_raster_plot(city: str, raster_type: str):
    if raster_type == "lcz":
        res = 100
    elif raster_type == "dtm":
        res = 30
    else:
        res = 10

    src = rasterio.open(f"../data/{raster_type}/{city}_{res}.tif")
    band = src.read(1)

    if raster_type == "lcz":
        lcz_colors = {
            1: "#8c0000",   # LCZ 1
            2: "#d10000",   # LCZ 2
            3: "#ff0000",   # LCZ 3
            4: "#bf4d00",   # LCZ 4
            5: "#ff6600",   # LCZ 5
            6: "#ff9955",   # LCZ 6
            7: "#faee05",   # LCZ 7
            8: "#bcbcbc",   # LCZ 8
            9: "#ffccaa",   # LCZ 9
            10: "#555555",  # LCZ 10
            11: "#006a00",  # LCZ A
            12: "#00aa00",  # LCZ B
            13: "#648525",  # LCZ C
            14: "#b9db79",  # LCZ D
            15: "#000000",  # LCZ E
            16: "#fbf7ae",  # LCZ F
            17: "#6a6aff",  # LCZ G (water)
        }
        cmap = ListedColormap(lcz_colors.values())

    elif raster_type == "bh":
        # mask knows which data has the noData value
        mask = src.read_masks(1)
        # so we set the noData value to 0 because there are no buildings there
        band[mask==0] = 0
        cmap = cmc.grayC_r

    elif raster_type == "dtm":
        cmap = ListedColormap(cmc.bukavu(np.linspace(0.5, 1.0, 256)))

    elif raster_type == "tcd":
        cmap = ListedColormap(cmc.bam(np.linspace(0.5, 1.0, 256)))

    norm = Normalize(vmin=np.nanmin(band), vmax=np.nanmax(band))
    # plt.imshow(band, cmap=lcm, norm=norm)
    plt.imshow(band, cmap=cmap, norm=norm if raster_type in ("dtm", "tcd") else None)
    plt.colorbar()
    plt.show()