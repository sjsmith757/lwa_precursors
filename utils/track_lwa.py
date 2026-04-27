# std libs
from calendar import monthrange
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union, overload, Set
from pathlib import Path
import logging

# third party
import xarray as xr
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

# custom libs
from .xr_tools import area_ave, get_area_grid, update_hist, get_num_cores
from .constants import RAD_EARTH
from .match_grid import match_grid

# specify the Paths to the dataset to track (FDIR),
# and where output data will be stored (ODIR)
ROOT = Path("MERRA2")
FDIR = ROOT / "6hrly" / "DTDTLWRCLR_VINT_UP"
ODIR = ROOT / "6hrly" / "LWA_COMPS"


# -----------------------------primary tracker functions--------------------------------
def find_centers(
    X: xr.DataArray,
    lwa_quantile: float = 0.9,
    area_thresh: float = 2.5e9,  # 2500 km^2 in m^2,
    lwa_thresh: Optional[Union[float, xr.DataArray]] = None,
) -> List[List[Tuple[float, float]]]:
    """
    identify the centers of large (vertically-integrated/ single-level) wave
    activity blobs. Since wave activity is an integrated field, we can just filter
    out small regions and small values and then identify the centers of mass. Then
    we just use the center of mass indices to figure out the coordinates

    Parameters
    ----------
    X : xarray.DataArray (time, lat, lon)
        wave activity field to identify centers in. Centers are identified
        independently of time/level, aka just as 2D lat/lon map. Data coordinates
        are expected to be named "time", "lat", and "lon". This should be just one
        hemisphere of data as wave activity has different magnitudes in different
        seasons
    lwa_quantile : float, optional
        the quantile to determine which wave activity blobs should be candidates for
        tracking, by default 0.9. Because wave activity is positive definite, only
        the upper bound is used
    area_thresh : float, optional
        the area a region must have in order to be considered a candidate for
        tracking in m^2, by default 2.5e9 m^2. There is some trade-off between the
        magnitude of the wave activity values required (set by lwa_quantile) and the
        area enclosed by a wave activity region above that magnitude, so these two
        parameters should be set jointly
    lwa_thresh : float or xr.DataArray, optional
        numeric threshold (in m/s) for determining which wave activity blobs should
        be candidates. If this is provided, lwa_quantile is ignored.

    Returns
    -------
    List[List[Tuple[float, float]]]
        Since there are different numbers of wave activity blobs to track at any
        given time in any given hemisphere, we have a "jagged" array which doesn't
        have any great data structures in python, so we just use a list of lists.
        There are ntime lists, each containing a list of lat/lon pairs of
        identified wave activity center coordinates
    """

    # get the threshold value based on the quantile and threshold
    if lwa_thresh is None:
        lwa_thresh = X.quantile(lwa_quantile).values.tolist()
    X_thresh = X.where(X > lwa_thresh, 0.0)
    # next compute the area of each grid cell so we can check to make sure the
    # candidate center is large enough
    dA = get_area_grid(X.lat, X.lon) * np.isfinite(X).astype("float")
    dA *= RAD_EARTH**2
    dA = dA.transpose("time", "lat", "lon")
    # now lets weight the wave activity field by area so the center of mass
    # calculations are correct
    cos_lat = np.cos(np.radians(X.lat))
    X_weight = X_thresh * cos_lat
    X_weight = X_weight.transpose("time", "lat", "lon")

    # now we'll get the centers of mass
    # first we need to identify all possibilities
    # if more connectedness desired...
    # ndimage.iterate_structure(ndimage.generate_binary_structure(2,1),2) -> 5x5 array
    struct = np.stack([np.zeros((3, 3)), np.ones((3, 3)), np.zeros((3, 3))], axis=0)
    lbls, nfeature = ndimage.label(X_weight.values, structure=struct)
    # now don't include the centers which aren't large enough in area
    feat_list = [
        ifeat
        for ifeat in range(1, nfeature + 1)
        if dA.where(lbls == ifeat, 0.0).sum() > area_thresh
    ]
    mass_list = ndimage.center_of_mass(X_weight.values, labels=lbls, index=feat_list)
    # unzip the indices provided
    time_inds, lat_inds, lon_inds = list(zip(*mass_list))
    # the indices computed here are floats, not ints, so we want to interpolate them
    # onto the grid to get centers that aren't fixed on the grid resolution
    lat_coord_list = np.interp(
        lat_inds, np.arange(X.lat.size, dtype="float"), X.lat.values
    )
    lon_coord_list = np.interp(
        lon_inds, np.arange(X.lon.size, dtype="float"), X.lon.values
    )
    # now we zip back up and leave off centers that couldn't be identified
    # (it returns some nans in some cases)
    coord_list: List[Tuple[float, float, float]] = list(
        zip(time_inds, lat_coord_list, lon_coord_list)
    )
    centers = [
        [
            (lat, lon)
            for (time, lat, lon) in coord_list
            if (np.round(time) == i if np.isfinite(time) else False)
        ]
        for i in range(X.time.size)
    ]
    return centers


def find_filtered_centers(X: xr.DataArray, **kwargs) -> List[List[Tuple[float, float]]]:
    """
    get a consolidated list of wave activity centroids based on the vertically
    integrated wave activity/wave activity at one level. This function takes the raw
    list of wave activity centers from find_centers and checks for centroids which were
    divided over the dateline/prime meridian to ensure only one center is computed in
    these cases. As such, this function is preferred to find_centers, although they
    accept the same parameters and return similar output

    Parameters
    ----------
    X : xarray.DataArray (time, lat, lon)
        wave activity field to identify centers in. Centers are identified independently
        of time/level, aka just as 2D lat/lon map. Data coordinates are expected to be
        named "time", "lat", and "lon". This should be just one hemisphere of data as
        wave activity has different magnitudes in different seasons. This function
        assumes that data is in the domain (-180,180) in longitude, data should be
        mapped to this domain before using, or this function will need to be modified
    lwa_quantile : float, optional
        the quantile to determine which wave activity blobs should be candidates for
        tracking, by default 0.9. Because wave activity is positive definite, only the
        upper bound is used
    area_thresh : float, optional
        the area a region must have in order to be considered a candidate for tracking
        in m^2, by default 2.5e9 m^2. There is some trade-off between the magnitude of
        the wave activity values required (set by lwa_quantile) and the area enclosed by
        a wave activity region above that magnitude, so these two parameters should be
        set jointly

    Returns
    -------
    List[List[Tuple[float, float]]]
        Since there are different numbers of wave activity blobs to track at any given
        time in any given hemisphere, we have a "jagged" array which doesn't have any
        great data structures in python, so we just use a list of lists. There are ntime
        lists, each containing a list of lat/lon pairs of identified wave activity
        center coordinates
    """

    # we need to make sure we are tracking in a periodic (lon) domain, so we will
    # search the field as it is, and the shifted version of the field by 180 degrees
    centers = find_centers(X, **kwargs)
    Xroll = X.roll(lon=X.lon.size // 2, roll_coords=True)
    Xroll = Xroll.assign_coords(lon=Xroll.lon.where(Xroll.lon > 0, Xroll.lon + 360.0))
    rollcenters = find_centers(Xroll, **kwargs)
    # we modified the coordinate system when shifting 180 degrees, now convert back to
    # old coords
    rollcenters = [
        [(lat, lon if lon < 180 else lon - 360) for (lat, lon) in rc]
        for rc in rollcenters
    ]
    # find any new centers which appeared after searching the rolled field
    # (these should be for regions that were split over the dateline)
    new_centers = [
        [pair for pair in rc if (not len(c)) or (not np.isclose(pair, c).any())]
        for rc, c in zip(rollcenters, centers)
    ]
    # find any old centers which went away after rolling (these centers were combined
    # into the new ones above)
    merged_centers = [
        [pair for pair in c if (not len(rc)) or (not np.isclose(pair, rc).any())]
        for rc, c in zip(rollcenters, centers)
    ]
    # In the rolled dataset, we now have a cut point at the prime meridian and we don't
    # want any new centers that appeared in that region, so just stick close to the
    # dateline
    new_centers = [
        [(lat, lon) for (lat, lon) in nc if np.abs(lon) > 120.0] for nc in new_centers
    ]
    merged_centers = [
        [(lat, lon) for (lat, lon) in mc if np.abs(lon) > 120.0]
        for mc in merged_centers
    ]
    # now update the centers with the new merged centers and remove old ones
    for c, mc, nc in zip(centers, merged_centers, new_centers):
        for pair in mc:
            c.remove(pair)
        c.extend(nc)

    return centers


def get_neighborhood(
    in_ds: xr.Dataset, in_c: Tuple[float, float], r: float
) -> xr.Dataset:
    """
    for a given group of variables contained in an xarray.Dataset object, hyperslab the
    variables in the vicinity/neighborhood of the identified wave activity centroid

    Parameters
    ----------
    in_ds : xarray.Dataset
        the Dataset of variables to hyperslab around the wave activity centroid. This
        functions assumes longitude is in the domain (-180,180). It also assumes
        coordinate names are 'lat' & 'lon'
    in_c : Tuple[float, float]
        the coordinates of a wave activity centroid
    r : float
        the physical distance in meters used to determine which grid points are in the
        neighborhood of the wave activity centroid

    Returns
    -------
    xarray.Dataset
        The hyper-slabbed Dataset with variable domains restricted to within the
        specified distance of the input wave activity centroid
    """

    # first off, we need to make sure that the requested neighborhood around the
    # centroid doesn't contain the dateline, or else we won't be able to hyperslab
    # it correctly. If it does contain the dateline, we need to shift the data
    # domain to (0,360)
    pole_bound = in_c[0] + np.sign(in_c[0]) * np.degrees(r / RAD_EARTH)
    if np.abs(in_c[1]) >= 180.0 - np.degrees(
        r / RAD_EARTH / np.cos(np.radians(pole_bound))
    ):
        ds = in_ds.roll(lon=in_ds.lon.size // 2, roll_coords=True)
        lon = ds.lon
        ds = ds.assign_coords(lon=lon.where(lon >= 0.0, lon + 360.0))
        lonc = in_c[1]
        c = (in_c[0], lonc + 360.0 if lonc < 0.0 else lonc)
    else:
        ds = in_ds
        c = in_c
    # let's compute the grid in physical coordinates from lat/lon
    ygrid = ds.lat * np.pi / 180.0 * RAD_EARTH
    xgrid = ds.lon * np.pi / 180.0 * RAD_EARTH * np.cos(ygrid / RAD_EARTH)
    pad = np.abs(ds.lat[1] - ds.lat[0]) * np.pi / 180.0 * RAD_EARTH
    # convert the input centers into physical coordinates
    yc = RAD_EARTH * c[0] * np.pi / 180.0
    xc = RAD_EARTH * c[1] * np.pi / 180.0 * np.cos(np.pi / 180.0 * ds.lat)
    # get a boolean mask of the region we want to keep, centered on the
    # wave activity centroid
    nbrhood = (np.abs(ygrid - yc) <= r + pad) & (np.abs(xgrid - xc) <= r + pad)
    # and finally hyperslab the Dataset, adding on the new physical coordinates that we
    # will be re-mapping/interpolating to
    regds = ds.where(nbrhood, drop=True)
    regds["X"] = (xgrid - xc).where(nbrhood, drop=True)
    regds["Y"] = (xr.ones_like(ds.lon) * (ygrid - yc)).where(nbrhood, drop=True)

    return regds


def find_regions(
    ds: xr.Dataset,
    centers: List[List[Tuple[float, float]]],
    r: float = 1e6,
    n: int = 25,
) -> xr.Dataset:
    """
    once we have the wave activity centroids, we want to take a group of variables
    (as an xarray.Dataset), subset/hyperslab that data in the vicinity/neighborhood of
    each centroid, and then interpolate/re-map from lat/lon coordinates onto a physical
    distance grid centered at the centroid. This will allow for easy compositing of the
    fields around the wave activity centroids no matter what latitude the centroid is
    actually at

    Parameters
    ----------
    ds : xarray.Dataset
        the data to composite around the various wave activity centroids. Data is
        expected to be in the domain (-180,180) in longitude, and to have coordinates
        named "lat" and "lon"
    centers : List[List[Tuple[float, float]]]
        A jagged array of wave activity centroids identified by find_filtered_centers
    r : float, optional
        the physical distance in meters used to determine the size of the area around a
        wave activity centroid to composite, by default 1e6
    n : int, optional
        the number of gridpoints in each new X/Y dimension, by default 25

    Returns
    -------
    xarray.Dataset
        A Dataset containing the original variables of the input Dataset, re-mapped onto
        physical coordinates around each wave activity center. A new dimension "ncent"
        is added to keep track of the multiple wave activity centroids that are
        identified at each time step. Because this is a "jagged" array (different
        numbers of centers at different time steps), ncent is set to the maximum number
        of centers identified for any timestep, and arrays are padded with NaN if they
        have fewer than the maximum number of centers for a timestep such that the
        resulting array is square
    """

    # for each center at each timestep/2D field, we need to get the neighborhood around
    # the centers, interpolate them into physical coordinates, and then merge everything
    # into a nice Dataset that we can write to a netCDF file for storage
    outds_list: List[xr.Dataset] = []
    for i, ic in enumerate(centers):
        ids = ds.isel(time=i).transpose("lon", "lat")
        ds_int_list: List[xr.Dataset] = []
        for j, (lat, lon) in enumerate(ic):
            # we start by just hyperslabbing the desired region around each centroid
            regds = get_neighborhood(ids, (lat, lon), r)

            da_int_list: List[xr.DataArray] = []
            out_coords: Dict[Hashable, Any] = {
                "time": [ds.time.values[i]],
                "ncent": [j + 1],
            }
            # unfortunately we have to use a low-level regridder and so we need to
            # iterate over each variable at each time step
            for dv in regds.data_vars:
                # we don't want to try to regrid the coordinates
                if dv not in ["X", "Y"]:
                    # since we are regridding from a spherical lat/lon grid to a locally
                    # Cartesian one, we have some "extra" lat/lon points in the
                    # hyperslab that aren't used - they are simply masked out as nans.
                    # This is because the area of each grid box decreases with latitude,
                    # so we need more lat/lon grid points at higher latitudes and fewer
                    # at lower latitudes, resulting in missing data patches of data we
                    # don't want. however, these NaNs will mess with our regridder, so
                    # we need to just flatten everything and get rid of the NaNs,
                    # providing only those data points we want to regrid to
                    # the regridder
                    da, regx, regy = xr.align(regds[dv], regds.X, regds.Y)
                    arr_flat = da.values.flatten()
                    da_nonan = arr_flat[np.isfinite(arr_flat)]
                    x_nonan = regx.values.flatten()[np.isfinite(arr_flat)]
                    y_nonan = regy.values.flatten()[np.isfinite(arr_flat)]
                    # here we construct the new coordinate system to regrid to
                    new_coord = np.linspace(-r, r, n)
                    newy, newx = np.meshgrid(new_coord, new_coord)
                    # regrid!
                    arr_int = griddata(
                        (x_nonan, y_nonan),
                        da_nonan,
                        (newx.flatten(), newy.flatten()),
                    ).reshape(1, 1, n, n)
                    # save it nicely as a DataArray
                    da_int = xr.DataArray(
                        arr_int,
                        dims=list(out_coords) + ["x", "y"],
                        coords={**out_coords, "x": new_coord, "y": new_coord},
                        name=dv,
                        attrs=ds[dv].attrs,
                    )
                    da_int_list.append(da_int)
            # we also want to save the coordinates of the wave activity centers that
            # provide the center of the new coordinate system for future reference
            latc = xr.DataArray(
                [[lat]],
                dims=list(out_coords),
                coords=out_coords,
                name="latc",
                attrs={"long_name": "composite center latitude"},
            )
            lonc = xr.DataArray(
                [[lon]],
                dims=list(out_coords),
                coords=out_coords,
                name="lonc",
                attrs={"long_name": "composite center longitude"},
            )
            ds_int_list.append(xr.merge(da_int_list + [latc, lonc]))
        # finally we stick everything back together
        if ds_int_list:
            ids_int = xr.concat(ds_int_list, dim="ncent")
            ids_int.time.encoding = ds.time.encoding
            outds_list.append(ids_int)
    # stick everything together and clean up the coordinates
    outds = xr.concat(outds_list, dim="time")
    outds = outds.reindex(time=ds.time)
    outds.x.attrs = {"long_name": "west-east distance from center", "units": "m"}
    outds.y.attrs = {"long_name": "south-north distance from center", "units": "m"}

    return outds


# -----------------------------------testing functions----------------------------------
@overload
def setup_test_data(
    *,
    cyc: bool = ...,
    ntime: int = ...,
    lwa_quantile: float = ...,
    area_thresh: float = ...,
) -> Tuple[xr.DataArray, List[List[Tuple[float, float]]]]: ...


@overload
def setup_test_data(
    *,
    cyc: bool = ...,
    ntime: int = ...,
    lwa_quantile: float = ...,
    area_thresh: float = ...,
    extravar: str,
) -> Tuple[xr.DataArray, List[List[Tuple[float, float]]], xr.DataArray]: ...


def setup_test_data(
    *,
    cyc: bool = True,
    ntime: int = 12,
    lwa_quantile: float = 0.9,
    area_thresh: float = 5e11,
    extravar: Optional[str] = None,
) -> Union[
    Tuple[xr.DataArray, List[List[Tuple[float, float]]]],
    Tuple[xr.DataArray, List[List[Tuple[float, float]]], xr.DataArray],
]:
    """
    get some test data and a list of wave activity centers for that data

    Parameters
    ----------
    cyc : bool, optional
        whether anticylonic (False) or cyclonic (True) wave activity is desired, by
        default True

    extravar : str, optional
        a variable to return from the dataset

    Returns
    -------
    xarray.DataArray
        the test wave activity data
    List[List[Tuple[float, float]]]]
        the jagged array of wave activity centers, with a list of center coordinates
        for each time
    xarray.Dataset
        the test data
    """

    nc = next(FDIR.glob("*200501.nc"))
    lnc = get_lwa_files(nc)

    ds = xr.open_mfdataset(lnc).isel(time=slice(0, ntime))
    # AeLp is positive PV anomaly, which is cyclonic in NH but anticylonic in SH,
    # likewise but opposite for AeLm. Because the magnitude of LWA depends on the sign
    # of the anomaly (since we tend to see sharper PV gradients in the vicinity of
    # cyclonic anomalies), we need to make sure the fields are either consistently
    # cyclonic or consistently anticyclonic
    if cyc:
        AeL = ds.AeLp_vint.where(ds.lat > 0.0, -ds.AeLm_vint)
    else:
        AeL = (-ds.AeLm_vint).where(ds.lat > 0.0, ds.AeLp_vint)
    AeL = AeL.rename("AeL")
    if AeL.lat[0] > AeL.lat[1]:
        AeL = AeL.sortby("lat")
    # we really only want to search the positive wave activity regions, negative ones
    # don't really have physical meaning but can mess with the tracking algorithm
    AeL = AeL.where(AeL > 0.0, 0.0)

    # make sure we look at the hemispheres separately, and also don't include too much
    # of the tropics as the QG assumption is questionable there in the first place.
    # likewise the polar regions can have some numerical issues due to the small areas
    # involved, so we'll stick with a sizeable midlatitude band
    NHcenters = find_filtered_centers(
        AeL.sel(lat=slice(25, 75)),
        area_thresh=area_thresh,
        lwa_quantile=lwa_quantile,
    )
    SHcenters = find_filtered_centers(
        AeL.sel(lat=slice(-75, -25)),
        area_thresh=area_thresh,
        lwa_quantile=lwa_quantile,
    )
    centers = [NHc + SHc for NHc, SHc in zip(NHcenters, SHcenters)]

    if extravar is not None:
        outvar = xr.open_dataset(nc).isel(time=slice(0, ntime))[extravar]
        if extravar == "qgpv":
            outvar = outvar.sel(lev=250, method="nearest")
        return AeL, centers, outvar
    else:
        return AeL, centers


def check_centers(
    *,
    cyc: bool = True,
    ntime: int = 12,
    lwa_quantile: float = 0.9,
    area_thresh: float = 5e11,
    plotvar: Optional[str] = None,
):
    """
    make some nice plots of the tracked wave activity centers so we can make sure the
    algorithm is working as intended

    Parameters
    ----------
    cyc : bool, optional
        whether to plot cyclonic (True) or anticylonic (False) wave activity,
        by default True
    plotvar: str, optional
        field to plot, wave activity field by default
    """

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from warnings import catch_warnings, simplefilter
    from shapely.errors import ShapelyDeprecationWarning

    # getting lots of annoying warnings here, probably a version mismatch issue
    # in my environment...
    simplefilter("ignore", category=ShapelyDeprecationWarning)

    if plotvar is None:
        AeLp, centers = setup_test_data(
            cyc=cyc, ntime=ntime, lwa_quantile=lwa_quantile, area_thresh=area_thresh
        )
        var = AeLp
        vmax = 0.5 * (
            AeLp.sel(lat=slice(-75, -25)).quantile(0.9).values
            + AeLp.sel(lat=slice(25, 75)).quantile(0.9).values
        )
        # vmax = 160.0
        vmin = 0.0
    else:
        AeLp, centers, var = setup_test_data(
            cyc=cyc,
            ntime=ntime,
            lwa_quantile=lwa_quantile,
            area_thresh=area_thresh,
            extravar=plotvar,
        )
        if plotvar == "U_vint":
            vmax = 50.0
            vmin = -25.0
        else:
            vmax = None
            vmin = None

    # we'll make a nice big multipanel plot to start off
    acstr = "cyclonic" if cyc else "anticyclonic"
    with catch_warnings():
        simplefilter("ignore")
        cs = var.plot.pcolormesh(
            row="time",
            col_wrap=4,
            subplot_kws={"projection": ccrs.Robinson()},
            cmap=plt.cm.cividis,
            transform=ccrs.PlateCarree(),
            vmax=vmax,
            vmin=vmin,
            extend="both",
            cbar_kwargs={
                "aspect": 40,
                "orientation": "horizontal",
                "label": f"vertically-integrated {acstr} LWA (m/s)",
            },
            figsize=(12.5, 2.5 * ntime // 4),
        )
        for ax in cs.axes.flatten():
            ax.coastlines(color="w")
    # then we'll go through and label the wave activity centers so we can see
    # how well we did
    for c, ax in zip(centers, cs.axes.flatten()):
        for pair in c:
            ax.scatter(
                pair[1],
                pair[0],
                s=30,
                c="r",
                marker="*",
                transform=ccrs.PlateCarree(),
                zorder=100,
            )

    plt.subplots_adjust(bottom=0.2)
    plt.show()

    return


def check_neighborhood(
    *,
    cyc: bool = True,
    r: float = 1e6,
    n: int = 25,
    lwa_quantile: float = 0.9,
    area_thresh: float = 5e11,
    plotvar: Optional[str] = None,
):
    """
    testing code to make sure the get_neighborhood function is working as expected

    Parameters
    ----------
    cyc : bool, optional
        whether to check anticyclonic (False) or cyclonic (True) wave activity
        centroids, by default True
    r : float, optional
        the physical distance radius around the centroid to use for re-mapping, in
        meters, by default 1e6
    """
    import matplotlib.pyplot as plt

    # get some test data to make sure the code is working properly
    # we'll test it on just a subset of the data
    if plotvar is None:
        A, centers = setup_test_data(
            cyc=cyc, ntime=4, lwa_quantile=lwa_quantile, area_thresh=area_thresh
        )
        comp_ds = find_regions(A.to_dataset(), centers, r=r, n=n)
        plotvar = "AeLp"
        var = A
    else:
        A, centers, var = setup_test_data(
            cyc=cyc,
            ntime=4,
            lwa_quantile=lwa_quantile,
            area_thresh=area_thresh,
            extravar=plotvar,
        )
        comp_ds = find_regions(var.to_dataset(), centers, r=r, n=n)
    # now we will plot the interpolated data against the original, hyperslabbed data
    _, axes = plt.subplots(3, 2, figsize=(8, 12))
    for i in range(3):
        comp_A = comp_ds[plotvar].isel(time=0, ncent=i)
        comp_A = comp_A.assign_coords(x=comp_A.x / 1000.0)
        comp_A = comp_A.assign_coords(y=comp_A.y / 1000.0)
        comp_A.x.attrs = {"long_name": "west-east distance", "units": "km"}
        comp_A.y.attrs = {"long_name": "south-north distance", "units": "km"}
        cent = centers[0][i]
        reg_A = get_neighborhood(var.isel(time=0).to_dataset(), cent, r)[plotvar]
        # get_neighborhood will shift the coordinate system if the center is too close
        # to the dateline, so we need to correct for this before plotting
        pole_bound = cent[0] + np.sign(cent[0]) * np.degrees(r / RAD_EARTH)
        if np.abs(cent[1]) >= 180.0 - np.degrees(
            r / RAD_EARTH / np.cos(np.radians(pole_bound))
        ):
            cent = (cent[0], cent[1] + 360 if cent[1] < 0.0 else cent[1])
        ax1, ax2 = axes[i]
        cs = reg_A.plot.contourf(ax=ax1, add_colorbar=False)
        # label the center
        ax1.scatter(cent[1], cent[0], s=20, c="k", marker="*")
        comp_A.plot.contourf(ax=ax2, levels=cs.levels, x="x", y="y")
        # add some axes and square it out (it should be square)
        ax2.set_aspect(1.0)
        ax2.axhline(0.0, lw=0.75, c="k")
        ax2.axvline(0.0, lw=0.75, c="k")
    plt.tight_layout()
    plt.show()
    return


# -------------------------------execution functions------------------------------------
def get_lwa_files(f: Path, input: bool = True) -> List[Path]:
    """
    helper func to get the wave activity file containing the same timesteps as
    the file to track

    Parameters
    ----------
    f : Path
        path of file to track

    input: bool, optional
        whether to get the input file to search (True), or a completed and saved output
        file (False), by default True

    Returns
    -------
    List[Path]
        list of paths of the wave activity files containing the same timesteps
    """

    time = f.name.split(".")[-2]
    code = "MERRA2_300" if int(time[:4]) < 2011 else "MERRA2_400"
    if input:
        ldir = ROOT / "6hrly/LVINT_UP"
    else:
        ldir = ODIR / "LVINT_UP"
    if len(time) < 8:
        days_in_month = monthrange(int(time[:4]), int(time[4:]))[1]
        return [
            ldir / f"{code}.inst6_3d_ana_Np.LVINT_UP.testnan.{time}{d+1:02d}.nc"
            for d in range(days_in_month)
        ]
    else:
        return [ldir / f"{code}.inst6_3d_ana_Np.LVINT_UP.testnan.{time}.nc"]


def get_lwa_thresh(f: Path, var: str) -> float:
    yrmn = f.name.split(".")[-2][:6]
    with xr.open_dataset(
        ROOT / "6hrly/MERRA2_300.inst6_3d_ana_Np.LWA_quantiles.2005_2019.nc"
    ) as qds:
        return qds[var].sel(time=f"{yrmn[:4]}-{yrmn[4:]}-01").values.tolist()


def track_file(f: Path):
    """
    for a given netCDF file, find the corresponding wave activity file, get the
    centroids for each time step in the wave activity file, subset the data file around
    the wave activity centroids, interpolate the new data subsets to a physical distance
    grid for easy compositing over the different wave activity blobs, and save the
    resulting dataset to a netCDF file. The resulting file will have the same variables
    as the initial file, plus variables latpc (latmc), lonpc (lonmc) which contain the
    coordinate pairs of the anticyclonic (cyclonic) wave activity centers. The file also
    contains a new dimension "ncent", which keeps track of the number of centroids which
    are identified at each time step. This is a "jagged" dimension, as there isn't a
    fixed number of centers identified at each time step. When a time step has fewer
    than the maximum number of centers, the array is padded with NaN values to square it
    out. If you are using xarray to process the data after tracking, you can easily
    check how many centers were identified at each time by doing latpc.notnull().sum
    ("ncent"). The tracking algorithms expect the following conventions for the data
    files:

    -   dimensions are named "time", "lat", "lon", and that data has at most 3
        dimensions. Dimensions in xarray can easily be renamed as e.g. dataset.rename
        (latitude="lat"). This algorithm isn't designed to work with 4 dimensional data
        right now as it uses an image processing technique to find and locate the wave
        activity centers. It could probably be extended with minimal effort, however
    -   the "lon" coordinate for the data should be in the domain (-180,180). It is
        much easier to shift/roll the data into a new domain with new coordinates than
        to modify the source code to handle a different domain, but in theory you could
        do the latter. Some pseudo-code using xarray to get started:

        ::code:
        rolled_dataset = dataset.assign_coords(
            lon=lon.where(lon < 180, lon - 360)
        ).roll(lon=lon.size//2, roll_coords=True)

    Parameters
    ----------
    f : Path
        Path object to the netCDF file to track
    """

    logging.info(f"tracking {f.name.split('.')[-2]}")
    # get the corresponding wave activity file for that timestep
    outlwafs = get_lwa_files(f, input=False)
    lwafs = get_lwa_files(f)
    if not all([olf.exists() for olf in outlwafs]):
        logging.debug("output files not found, finding centers")
        for outlwaf, lwaf in zip(outlwafs, lwafs):
            if not outlwaf.exists():
                # open the data, make sure all data is positive/sorted
                lwa = xr.open_dataset(lwaf)[["AeLp_vint", "AeLm_vint"]]
                if lwa.lat[0] > lwa.lat[1]:
                    lwa = lwa.sortby("lat")
                lwa["AeLm_vint"] *= -1
                lwa = lwa.where(lwa > 0.0, 0.0)
                # get the centers for Southern hemisphere
                acyc_centers, cyc_centers = [
                    find_filtered_centers(
                        lwa[var].sel(lat=slice(-75, -25)),
                        area_thresh=1e12,
                        # lwa_quantile=0.9,
                        lwa_thresh=get_lwa_thresh(lwaf, var),
                    )
                    for var in ["AeLp_vint", "AeLm_vint"]
                ]
    else:
        centds = xr.open_mfdataset(outlwafs)
        # in case we lost any times, i.e. they didn't have any detected centers,
        # make sure we add them back. this is because find_filtered_centers returns
        # an empty list when no centers are found and we want to have the same data
        # structure when reading from file
        time = xr.open_mfdataset(lwafs).time
        centds = centds.reindex(time=time)
        acyc_centers = [
            [
                (
                    float(centds.latpc.sel(ncent=c, time=t).values),
                    float(centds.lonpc.sel(ncent=c, time=t).values),
                )
                for c in centds.ncent
                if centds.latpc.sel(ncent=c, time=t).notnull()
            ]
            for t in centds.time
        ]
        cyc_centers = [
            [
                (
                    float(centds.latmc.sel(ncent=c, time=t).values),
                    float(centds.lonmc.sel(ncent=c, time=t).values),
                )
                for c in centds.ncent
                if centds.latmc.sel(ncent=c, time=t).notnull()
            ]
            for t in centds.time
        ]

    # now open and preprocess the data for tracking, making sure it is southern
    # hemisphere, also toss out stuff that isn't 3D (time/lat/lon)
    ds = xr.open_dataset(f)
    ds3d = ds[[var for var in ds.data_vars if "lon" in ds[var].dims]]
    ds3d = ds3d.sortby("lat").sel(lat=slice(-90, 0.0))
    # some variables shouldn't be tracked in both cyclonic/anticyclonic fields
    # and some should, depending on the file
    # so we'll split into two datasets
    if FDIR.name == "LVINT_UP":
        cds = ds3d[
            [dv for dv in ds.data_vars if "AeLm" in str(dv)]
            + ["etam_vint", "qevm_vint", "EMFC_vint", "EHFC_vint"]
        ]
        acds = ds3d[
            [dv for dv in ds.data_vars if "AeLp" in str(dv)]
            + ["etap_vint", "qevp_vint", "EMFC_vint", "EHFC_vint"]
        ]
        acds = acds.rename({"EMFC_vint": "EMFCp_vint", "EHFC_vint": "EHFCp_vint"})
        cds = cds.rename({"EMFC_vint": "EMFCm_vint", "EHFC_vint": "EHFCm_vint"})
        for dv in ["EMFC_vint", "EHFC_vint"]:
            cds[dv.replace("_vint", "m_vint")].attrs = {
                "long_name": "cyclonic " + ds3d[dv].attrs.get("long_name", dv),
                "units": ds3d[dv].attrs.get("units", ""),
            }
            acds[dv.replace("_vint", "p_vint")].attrs = {
                "long_name": "anticyclonic " + ds3d[dv].attrs.get("long_name", dv),
                "units": ds3d[dv].attrs.get("units", ""),
            }
    # other fields it's nice to see both the raw field and the zonally-anomalous field,
    # since cyclonic and anticylonic wave activity occur at preferrentially different
    # latitudes, and we may want to composite the anomalous fields. And of course, we
    # still need to split into two datasets
    else:
        if ("PV" not in FDIR.name) and ("EP_" not in FDIR.name):
            ds3d = match_grid(
                ds3d.to_array("variable"), sort_dims=["lat", "lon"]
            ).to_dataset("variable")
        ds3d_anom = ds3d - area_ave(ds3d, dims="lon")
        for dv3 in ds3d_anom.data_vars:
            ds3d_anom = ds3d_anom.rename({dv3: f"{dv3}_anom"})
            if dv3 in ds3d.data_vars:
                ds3d_anom[f"{dv3}_anom"].attrs = {
                    "long_name": "zonally anomalous "
                    + ds3d[dv3].attrs.get("long_name", dv3),
                    "units": ds3d[dv3].attrs.get("units", ""),
                }
        ds3d = xr.merge([ds3d, ds3d_anom])
        cds = ds3d.rename({dv: f"{dv}m" for dv in ds3d.data_vars})
        acds = ds3d.rename({dv: f"{dv}p" for dv in ds3d.data_vars})
        for dv, da in ds3d.data_vars.items():
            if dv + "m" in cds.data_vars:
                cds[dv + "m"].attrs = {
                    "long_name": "cyclonic " + da.attrs.get("long_name", dv),
                    "units": da.attrs.get("units", ""),
                }
            if dv + "p" in acds.data_vars:
                acds[dv + "p"].attrs = {
                    "long_name": "anticyclonic " + da.attrs.get("long_name", dv),
                    "units": da.attrs.get("units", ""),
                }

    logging.debug("finding regions")
    # now we go ahead and get the regional data around each center, and we'll recombine
    # into one dataset for saving as a netCDF
    ccomp_ds = find_regions(cds, cyc_centers, r=1.5e6, n=35).rename(
        lonc="lonmc", latc="latmc"
    )
    accomp_ds = find_regions(acds, acyc_centers, r=1.5e6, n=35).rename(
        lonc="lonpc", latc="latpc"
    )
    comp_ds = xr.merge([ccomp_ds, accomp_ds])
    # tweak/copy attrs and save
    for dv in comp_ds.data_vars:  # type: ignore
        if dv in acds.data_vars:
            comp_ds[dv].attrs = acds[dv].attrs
        elif dv in cds.data_vars:
            comp_ds[dv].attrs = cds[dv].attrs
    ncattrs = ds.attrs
    update_hist(ncattrs, "lwa tracked")
    comp_ds.attrs = ncattrs

    compnc = ODIR / f.parent.name / f.name
    compnc.parent.mkdir(exist_ok=True)
    comp_ds.to_netcdf(compnc)
    logging.info(f"tracked {f.name.split('.')[-2]}!")

    return


def parallel_safe_track_file(f: Path):
    try:
        track_file(f)
    except Exception as e:
        logging.warning(f"{f.name} failed with error {e}")
    return


if __name__ == "__main__":
    from multiprocessing import Pool
    from tqdm import tqdm

    # helpful test functions to play with parameters
    # check_centers(
    #     cyc=False, ntime=8, lwa_quantile=0.9, area_thresh=1e12
    # )  # , plotvar="qgpv")
    # check_neighborhood(cyc=False, r=1.5e6, area_thresh=1e12, n=35, plotvar="T_vint")
    # exit()

    # we can do this in serial mode or in parallel
    parallel = True
    # sometimes we want to run the code and just have it figure out what files are left
    # (False), but other times we want it to recompute everyting, even if it's been done
    # before (True)
    overwrite = False

    ODIR.mkdir(exist_ok=True)

    # see what's been done
    donefiles: Set[Path]
    if overwrite:  # nothing if we're overwriting
        donefiles = set()
    else:
        donefiles = set(FDIR / f.name for f in (ODIR / FDIR.name).glob("*.nc"))
    # get a list of what's left to do
    dofiles = sorted(set(FDIR.glob("*.nc")) - donefiles)

    # if running in parallel, we'll do it as an embarrassingly parallel problem using
    # multiprocessing.Pool, then we'll wrap it in a nice progress bar
    if parallel:
        logging.basicConfig(level=logging.WARNING)
        with Pool((get_num_cores() - 1) // 2) as mypool:
            list(
                tqdm(
                    mypool.imap_unordered(parallel_safe_track_file, dofiles),
                    total=len(dofiles),
                    ascii=True,
                )
            )
    # otherwise, it's just one at a time (actually just one for testing)
    else:
        logging.basicConfig(level=logging.INFO)
        logging.info(f"{len(dofiles)} remaining")

        for f in dofiles[:1]:
            track_file(f)
