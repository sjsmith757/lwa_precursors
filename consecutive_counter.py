# coding: utf-8
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def compute_consecutive_counts(arr: xr.DataArray, thresh: xr.DataArray) -> xr.DataArray:
    """
    compute the number of consecutive times a metric exceeds a threshold, or how many
    times it is currently exceeding it, at any given timestep and location

    Parameters
    ----------
    arr : xr.DataArray
        the metric array, which should contain dimension "time"
    thresh : xr.DataArray
        the threshold for the metric

    Returns
    -------
    xr.DataArray
        the consecutive counts for exceeding the threshold. The array has the same shape
        as the input `arr`, but the data is replaced with the number of consecutive times
        the threshold is exceeded. Each time step has a count equal to the duration of
        time spent above the threshold for the entire duration

        ex.
        >>> arr = [1 2 3 4 5 4 3 2 1 4 3 1 3]
        >>> thresh = 3
        >>> compute_consecutive_counts(arr, thresh)
        [0 0 5 5 5 5 5 0 0 2 2 0 1]

        run `basic_test_consecutive_counts` to see the example in action

    """

    # number of times exceeding the threshold
    counts = arr >= thresh
    # total sum - cumulative sum = how many counts are remaining (+1 to include
    # the current count)
    succeeding_counts = (counts.sum("time") + 1 - counts.cumsum("time")).where(
        counts, 0
    )
    # the first time we exceed the threshold (counts jumps from 0 to 1)
    first_in_group = counts.astype("int").diff("time") == 1
    #
    group_starts = succeeding_counts.where(first_in_group, drop=True)
    preceding_group_counts = (
        group_starts.reindex(time=arr.time).shift(time=-1).bfill("time").fillna(0)
    )
    counts_remaining = (succeeding_counts - preceding_group_counts).where(counts, 0)
    first_in_group = xr.concat(
        [xr.ones_like(counts.isel(time=0)), first_in_group], "time"
    )
    return (
        counts_remaining.where(first_in_group)
        .reindex(time=arr.time)
        .ffill("time")
        .where(counts, 0)
        .astype("int")
    )


def basic_test_consecutive_counts():
    # a simple example where we can figure out what the answer should be
    arr_list = [1, 2, 3, 4, 5, 4, 3, 2, 1, 4, 3, 1, 3]
    thresh = 3
    # we need to add a time coordinate since compute_consecutive_counts is designed to
    # work with xarray objects
    arr = xr.DataArray(
        arr_list,
        coords={"time": xr.date_range("2024-01-01", periods=len(arr_list), freq="d")},
        dims=("time",),
    )
    print(compute_consecutive_counts(arr, thresh))


def test_consecutive_counts():
    # now we test on real data
    # first load it all. including part of august to capture events which occur toward
    # the end of the month
    jul82files = sorted(Path("MERRA2/daily/TMAX").glob("*19820[7-8]??.nc"))[:38]
    JAthresh = xr.open_mfdataset("MERRA2/stats/TMAXP95/*0[7-8]??.nc").TMAXP95
    JAthresh = JAthresh.sel(lat=slice(30, 70))
    JAthresh.load()
    tmax = xr.open_mfdataset(jul82files).TMAX.sel(lat=slice(30, 70), lon=slice(-30, 40))
    tmax.load()
    # let's use the july threshold in july and the august one in august. In order to do
    # that, we have to add a time dimension corresponding to the month, and then populate
    # it with the same values for each day in the month
    JAthresh_daily = (
        JAthresh.squeeze()
        .rename(month="time")
        .assign_coords(time=[tmax.time[0].values, tmax.time[31].values])
        .reindex(time=tmax.time)
        .ffill("time")
    )
    # compute the counts
    consec_counts = compute_consecutive_counts(tmax, JAthresh_daily).transpose(
        "time", ...
    )
    # pick a day and see if the maximum duration event is done correctly
    t0 = 21
    # get location of the max
    max_iloc = consec_counts[t0].argmax(["lat", "lon"])
    # we can't use the dictionary as is because it has some extra coordinates,
    # (it returns xarray.Variables instead of just integers), so strip that off
    iloc_dict = {k: v.values for k, v in max_iloc.items()}
    # the count information only tells us we are in an event lasting a certain number of
    # days, not where we are in the event, so we have to find the start
    max_counts = consec_counts.isel(iloc_dict)
    max_count = max_counts[t0].values
    max_start = np.abs(max_counts.diff("time") - max_count).argmin().values
    # now compare the calculated duration of the event to one day before exceeding the
    # threshold to one day after the supposed end. We should see a 0 followed by the
    # number of ones computed as the duration, followed by another 0. Note this may not
    # work exactly if we are at the beginning or end of the time period, we may miss the
    # initial/final 0
    print(max_count)
    print(
        (tmax >= JAthresh_daily)
        .isel(iloc_dict)
        .astype("int")[max_start : max_start + max_count + 2]
    )

    # now let's look at some posssible heat wave events. We want to keep all times that
    # have an event that is lasting longer than 7 days, and we only want to keep these
    # times if they have more than a few grid points that are lasting longer than 7 days
    heatwave_counts = (
        consec_counts.where((consec_counts >= 7).sum(["lat", "lon"]) > 5, drop=True)
        .squeeze(drop=True)
        .sel(time="1982-07")
        .reset_coords(drop=True)
    )
    # compute how warm TMAX is compared to the average temperature of the month, so we
    # can see if there is a true heat wave
    tmax_anom = (
        (tmax.sel(time=heatwave_counts.time) - tmax.mean("time"))
        .squeeze(drop=True)
        .sel(time="1982-07")
        .reset_coords(drop=True)
    )

    # now plot
    # figure out a good grid layout
    ncols = 4
    nrows = (heatwave_counts.time.size + ncols - 1) // ncols
    fig, axs = plt.subplots(
        nrows, ncols, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(14, 7)
    )
    ax: plt.Axes
    # plot the temperature anomalies as shading and identified "heat wave" regions as
    # contours, and make it look nice
    for i, ax in enumerate(axs.flatten()):
        if i < tmax_anom.time.size:
            cs = tmax_anom.isel(time=i).plot(
                ax=ax, add_colorbar=False, vmin=-10, vmax=10, cmap=plt.cm.RdBu_r
            )
            heatwave_counts.isel(time=i).plot.contour(
                levels=[2, 5, 8], ax=ax, add_labels=False, colors="k"
            )
            ax.coastlines(color="grey")
            plt.xlabel("")
            plt.ylabel("")
        else:
            ax.remove()
    fig.colorbar(
        cs,
        ax=axs.flatten()[: tmax_anom.time.size],
        label="TMAX anomaly [K]",
        orientation="horizontal",
        extend="both",
        aspect=30,
    )
    fig.subplots_adjust(bottom=0.25, top=0.95, left=0.03, right=0.97)
    plt.show()


# this line is necessary to ensure our tests don't run if we import the function into
# another notebook
if __name__ == "__main__":
    basic_test_consecutive_counts()
    # test_consecutive_counts()
