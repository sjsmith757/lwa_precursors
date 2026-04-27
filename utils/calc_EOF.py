from functools import partial
from typing import (
    List,
    Optional,
    Union,
    Tuple,
    TypedDict,
    Dict,
    Literal,
    Callable,
    Any,
    Collection,
    Hashable,
)
import numpy as np
import xarray as xr
from scipy import signal
from scipy.integrate import simpson
from warnings import warn
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import warnings

# import os


class OpenParams(TypedDict, total=False):
    """
    dummy class for statically typing dictionary-style inputs to xr.open_mfdataset
    """

    chunks: Optional[Dict[str, Optional[Union[str, int]]]]
    join: Literal["outer", "inner", "left", "right", "exact", "override"]
    concat_dim: str
    coords: str
    combine: Literal["by_coords", "nested"]
    data_vars: Union[List[str], Literal["all", "minimal", "different"]]
    use_cftime: bool
    parallel: bool
    compat: Literal[
        "identical", "equals", "broadcast_equals", "no_conflicts", "override"
    ]
    preprocess: Callable


def calc_EOF(
    X: xr.DataArray,
    n: Optional[int] = None,
    return_decomp: bool = False,
    detrend: bool = True,
    anomalousX: bool = False,
    normalize: bool = False,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    compute the EOFs for a given DataArray. does detrending and deseasonalizing, if
    requested, as well as mass/area weighting

    Parameters
    ----------
    X : xr.DataArray
        field for computing EOFs
    n : int, optional
        number of principal components/EOFs to keep, by default None
    return_decomp : bool, optional
        whether to return the raw SVD or the weighted EOF/timeseries such that the
        anomalies of X can be reconstructed as the product of the EOFs and timeseries,
        by default False
    detrend : bool, optional
        whether to perform linear detrending with scipy.signal.detrend
    anomalousX : bool, optional
        if true, X is assumed to be anomalies and no anomalies will be calculated.
        Default False, anomalies will be calculated against the monthly mean X
    normalize : bool, optional
        whether to normalize the basis vectors and the PC timeseries. Default False

    Returns
    -------
    PCs : xr.DataArray
        the principal component timeseries
    EOFs : xr.DataArray
        the EOF spatial bases, weighted by singular values if return_decomp is False
    vrnce : xr.DataArray
        variance explained by each EOF, or singular values if return_decomp is True
    """

    # wrapper to simplify apply_ufunc call
    def svd_wrap(
        X: np.ndarray, return_decomp: bool = return_decomp, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if "n" in kwargs:
            n = kwargs.pop("n")
        else:
            n = None
        if "full_matrices" in kwargs:
            kwargs.pop("full_matrices")
        U, s, VH = np.linalg.svd(X, full_matrices=False, **kwargs)
        if return_decomp:
            return (
                U[:, slice(None, n)],
                VH[slice(None, n), :],
                s[slice(None, n)],
            )
        else:
            vrnce = s * s / (s * s).sum()
            return (
                U[:, slice(None, n)],
                (np.diag(s) @ VH)[slice(None, n), :],
                vrnce[slice(None, n)],
            )

    # calc anomaly and detrend
    if anomalousX:
        Xanom = X.fillna(0.0)
    else:
        Xmean: xr.DataArray = X.groupby("time.month").mean("time")
        Xanom = (X.groupby("time.month") - Xmean).fillna(0.0)

    if detrend:
        X_detrend: xr.DataArray = xr.apply_ufunc(
            signal.detrend, Xanom.transpose(..., "time")
        )
    else:
        X_detrend = Xanom.transpose(..., "time")
    sp_coords = [d for d in X_detrend.dims if d != "time"]

    # weight and compute EOFs
    weights = np.cos(np.pi / 180.0 * X["lat"]).astype(X.dtype)
    if "lev" in X_detrend.coords:
        if X["lev"].size > 1:
            Xlevs = X["lev"].sortby("lev", ascending=False)
            weights = weights * np.abs(Xlevs.diff("lev", label="upper")).astype(X.dtype)
    Xweight = X_detrend * np.sqrt(weights)
    X_flat = Xweight.stack({"grid": sp_coords})
    svd_out: Tuple[xr.DataArray, xr.DataArray, xr.DataArray] = xr.apply_ufunc(
        svd_wrap,
        X_flat,
        input_core_dims=[["time", "grid"]],
        output_core_dims=[["time", "pcrank"], ["pcrank", "grid"], ["pcrank"]],
        exclude_dims=set(["time", "grid"]),
        kwargs={"n": n},
    )
    PCs, EOF_flat, vrnce = svd_out

    PCs["time"] = X_flat["time"]
    EOF_flat = EOF_flat.assign_coords(grid=X_flat.grid.indexes["grid"])
    EOFs = EOF_flat.unstack() / np.sqrt(weights)
    if normalize:
        EOF_norms = (EOFs.stack({"grid": sp_coords})).reduce(np.linalg.norm, "grid")
        EOFs = EOFs / EOF_norms
        PCs = PCs * EOF_norms
    # PCstd = PCs.std("time")
    # stdize PCs, preserve product
    # PCs /= PCstd
    # EOFs *= PCstd

    pcrank = xr.DataArray(
        np.arange(PCs.pcrank.size),
        dims=["pcrank"],
        coords={"pcrank": np.arange(PCs.pcrank.size)},
        name="pcrank",
    )
    pcrank.attrs = {"long_name": "sorted number of PCs", "units": ""}
    PCs.name = "PC"
    PCs.attrs = {"long_name": "principal component timeseries", "units": "std devs"}
    PCs = PCs.assign_coords({"pcrank": pcrank})
    EOFs.name = "EOF"
    EOFs.attrs = {"long_name": "empirical orthogonal function spatial basis"}
    EOFs = EOFs.assign_coords({"pcrank": pcrank})
    vrnce.name = "vrnce"
    vrnce.attrs = {"long_name": "variance explained by EOF", "units": ""}
    vrnce = vrnce.assign_coords({"pcrank": pcrank})

    return PCs, EOFs, vrnce


def get_area_grid(lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    """
    helper func to generate area of grid boxes on original grid, specified by
    lat/lon

    Parameters
    ----------
    lat : xr.DataArray
        latitudes of input grid
    lon : xr.DataArray
        longitudes of input grid

    Returns
    -------
    xr.DataArray
        area grid
    """

    # lat bound checking
    dlat = np.abs(lat[1].values - lat[0].values)
    minlat = lat.min().values
    maxlat = lat.max().values
    slatb = minlat - dlat if minlat - dlat > -90.0 else -90.0
    nlatb = maxlat + dlat if maxlat + dlat < 90.0 else 90.0
    # lon bound checking
    dlon = np.abs(lon[1].values - lon[0].values)
    minlon = lon.min().values
    maxlon = lon.max().values
    elonb = minlon - dlon
    wlonb = maxlon + dlon

    # boundaries
    ext_lon = xr.concat(
        [xr.DataArray([elonb, wlonb], dims="lon", coords={"lon": [elonb, wlonb]}), lon],
        "lon",
    ).sortby("lon")
    lonb: xr.DataArray = 0.5 * (
        ext_lon[:-1] + ext_lon[1:].assign_coords({"lon": ext_lon[:-1]})
    )
    ext_lat = xr.concat(
        [xr.DataArray([slatb, nlatb], dims="lat", coords={"lat": [slatb, nlatb]}), lat],
        "lat",
    ).sortby("lat")
    latb: xr.DataArray = 0.5 * (
        ext_lat[:-1] + ext_lat[1:].assign_coords({"lat": ext_lat[:-1]})
    )

    # weights
    sin_latb: Any = np.sin(np.radians(latb))
    dX = np.radians(np.abs(lonb.diff("lon", label="upper")))
    dY = np.abs(sin_latb.diff("lat", label="upper"))
    return dX * dY


def area_ave(
    arr: Union[xr.Dataset, xr.DataArray],
    dims: Union[Collection[Hashable], str] = ("lat", "lon"),
    filled: bool = False,
    keep_attrs: bool = False,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    computes areal average of an xarray object over the given dimensions

    Parameters
    ----------
    arr : xr.DataArray or xr.Dataset
        array or dataset to be averaged
    dims : list of hashable, optional
        coordinates to average over, by default ["lat", "lon"]
    filled : bool, optional
        whether nan values should be replaced by 0, by default False
    keep_attrs : bool, optional
        whether to keep the objects original attrs, by default False

    Returns
    -------
    xr.DataArray or xr.Dataset
        averaged array/dataset
    Optionally xr.DataArray
        area weights used for averaging
    """

    # be careful when passed data is not floats, may not behave as expected
    if isinstance(arr, xr.Dataset):
        modarr = arr[[d for d in arr.data_vars][0]]
        arrdims = list(modarr.dims)
        arrshape = modarr.shape
        arrtype = modarr.dtype
        for da in arr.data_vars:
            assert np.all(list(arr[da].dims) == arrdims), (
                "to perform area averaging on dataset objects,"
                " all arrays must have same dimensions"
            )
            assert np.all(arr[da].shape == arrshape), (
                "to perform area averaging on dataset objects,"
                " all arrays must have same grid"
            )
    else:
        modarr = arr
        arrtype = arr.dtype

    dA = (get_area_grid(arr.lat, arr.lon) * np.isfinite(modarr)).astype(arrtype)

    arr_bar = arr.weighted(dA).mean(dim=dims, skipna=True, keep_attrs=keep_attrs)
    if filled:
        arr_bar = arr_bar.where(np.isfinite(arr_bar), other=0.0)

    return arr_bar


def vint(
    da: Union[xr.Dataset, xr.DataArray],
    ps: Optional[xr.DataArray] = None,
    pascals: bool = False,
    skipna: bool = True,
    simp_method: bool = True,
) -> Union[xr.Dataset, xr.DataArray]:
    """
    vertically-integrate an xarray object with optional weighting by surface
    pressure/mass

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        xarray data to integrate, with expected vertical coordinate lev
    ps : xr.DataArray, optional
        surface pressure data, by default None. Should be same units as lev
    pascals : bool, optional
        whether data is in pascals, by default False. Default assumes vertical
        coordinate is in hPa.
    skipna : bool, optional
        whether to treat NaN values as massless/zero, by default True
    simp_method: bool, optional
        whether to use the 4th order simpson scheme (true) or 2nd order trapezoidal
        scheme (False)

    Returns
    -------
    xr.DataArray or xr.Dataset
        vertically-integrated (mass-weighted) data
    """

    assert "lev" in da.coords, "Must have vertical coordinate (expected name lev)"
    surface_is_bottom = ps is not None
    if surface_is_bottom:
        assert np.all(
            sorted(map(str, ps.coords))  # type: ignore
            == sorted(c for c in map(str, da.coords) if c != "lev")
        ), "PS coords should match data coords, excluding lev"

    da_inc = da.copy(deep=False)
    # integrate top down (0 to ps)
    da_inc = da_inc.sortby("lev", ascending=True)
    # integration multiplies the coordinate values by the data values, which can
    # artificially inflate the precision of the data (i.e., lev is double but da is
    # single). We fix the precision of the coordinate values to those of the data
    dadtype = (
        da[list(da.data_vars)[0]].dtype if isinstance(da, xr.Dataset) else da.dtype
    )
    da_inc = da_inc.assign_coords(lev=da_inc["lev"].astype(dadtype))

    da_inc, ps = validate_units(da_inc, ps, pascals=pascals)

    # xarray forces skipna=False on integrate method,
    # so need to set nan to zero before integrating if not desired behaviour
    if skipna:
        da_inc = da_inc.fillna(0.0)
    if simp_method:
        dint = xr.apply_ufunc(
            simpson, da_inc, da_inc["lev"], input_core_dims=[["lev"], ["lev"]]
        )
    else:
        dint = da_inc.integrate("lev")

    pbot = da_inc.lev.values[-1]
    ptop = da_inc.lev.values[0]
    if surface_is_bottom:
        col_mass = ps.where(ps <= pbot, pbot).squeeze(drop=True) - ptop  # type:ignore
    else:
        col_mass = pbot - ptop

    return dint / col_mass


def validate_units(
    da: Union[xr.Dataset, xr.DataArray],
    ps: Optional[xr.DataArray] = None,
    pascals: bool = False,
    pa_thresh: float = 1200.0,
) -> Tuple[Union[xr.Dataset, xr.DataArray], Optional[xr.DataArray]]:
    """
    ensure a 3D DataArray/Dataset and surface pressure have the same units for vertical
    integration

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        the array to be integrated, should have one coordinate named "lev"
    ps : xr.DataArray, optional
        the surface pressure, by default None
    pascals : bool, optional
        whether the data is expected to be in Pa (True) or hPa (False), by default False
    pa_thresh : float, optional
        the value to distinguish between hPa and Pa, by default 1200.0

    Returns
    -------
    da : xr.DataArray or xr.Dataset
        input array with modified vertical coordinate (if necessary)
    ps : xr.DataArray, optional
        input surface pressure with modified units (if present)
    """
    surface_is_bottom = ps is not None
    if surface_is_bottom:
        ps_max = ps.max()  # type: ignore
    pbot: float = da["lev"].values[-1]  # type: ignore
    if pascals:
        hpa_warning = "expected data in Pa but data appears to be hPa, changing units"
        if pbot < pa_thresh:
            warnings.warn(hpa_warning, category=RuntimeWarning, stacklevel=2)
            da = da.assign_coords(lev=da["lev"].values * 100.0)  # type: ignore
        if surface_is_bottom:
            if ps_max < pa_thresh:
                warnings.warn(hpa_warning, category=RuntimeWarning, stacklevel=2)
                ps = ps * 100.0  # type: ignore
    else:
        pa_warning = "expected data in hPa but data appears to be Pa, changing units"
        if pbot > pa_thresh:
            warnings.warn(pa_warning, category=RuntimeWarning, stacklevel=2)
            da = da.assign_coords(lev=da["lev"].values / 100.0)  # type: ignore
        if surface_is_bottom:
            if ps_max > pa_thresh:
                warnings.warn(pa_warning, category=RuntimeWarning, stacklevel=2)
                ps = ps / 100.0  # type: ignore
    return da, ps


def sel_season(
    da: Union[xr.Dataset, xr.DataArray], ssn: str
) -> Union[xr.Dataset, xr.DataArray]:
    """
    susbsets a timeseries stored as an xarray object to a specific season specified by a
    season code utilizing the first letters of each month in the season, e.g. DJF for
    December-February or AMJJAS for April-September

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        timeseries of data to subset along dimension assumed to be named 'time'
    ssn : str
        season code made up of the first letter of consecutive months

    Returns
    -------
    xr.DataArray or xr.Dataset
        subset timeseries with values only during the requested season
    """

    if len(ssn) < 2:
        warn("using single-month codes is ambiguous and should be avoided")
    ssn_start = (2 * "JFMAMJJASOND").find(ssn.upper())
    if ssn_start != -1:
        month_inds = [(i + ssn_start) % 12 + 1 for i in range(len(ssn))]
        return da.where(
            xr.apply_ufunc(np.in1d, da["time.month"], month_inds), drop=True
        )
    else:
        if ssn[:3].lower() != "ann":
            warn("season was not recognized, returning year-round data", stacklevel=2)
        return da


def get_num_pcs(vrnce: xr.DataArray, tol: float) -> int:
    cumvrnce = vrnce.cumsum()
    if (cumvrnce > tol).any():
        return (cumvrnce - tol).where(cumvrnce > tol).argmin().values.tolist() + 1
    else:
        warnings.warn(
            "the cumulative variance does not reach the specified tolerance",
            RuntimeWarning,
            2,
        )
        return vrnce.pcrank.size + 1


def open_file(f: Path, var: str = "LVINT_GEO") -> xr.Dataset:
    dvar_lookup: dict[Hashable, list[Hashable]] = {
        "LVINT_GEO": ["AeLp", "AeLm"],
        "QGPV_GEO_VINT": ["qgpv"],
        "UGEO_VINT": ["UG"],
    }

    with xr.open_dataset(f) as ds:
        if var in dvar_lookup:
            ds = ds[[f"{v}_vint" for v in dvar_lookup[var]]]
        subds = ds.sel(lat=slice(20, 80), lon=slice(-80, 40))
    # if var == "LVINT_GEO":
    #     subds = (subds["AeLp_vint"] - subds["AeLm_vint"]).to_dataset()
    return subds


def load_data(var: str = "LVINT_GEO", anom: bool = True) -> xr.DataArray:
    path = Path("MERRA2/daily") / var
    all_files = sorted(path.glob("*.nc"))
    with Pool(4) as p:
        data = xr.concat(
            list(
                tqdm(
                    p.imap(partial(open_file, var=var), all_files),
                    ascii=True,
                    total=len(all_files),
                )
            ),
            "time",
        )
    if anom:
        return data.groupby("time.month") - data.groupby("time.month").mean("time")
    else:
        return data


if __name__ == "__main__":
    anom_ds = load_data("UGEO_VINT")

    print("computing EOFs")
    for v in ["UG"]:  # , "AeLm"]:
        pcds = xr.merge(
            calc_EOF(
                anom_ds[f"{v}_vint"],
                n=550,
                detrend=False,
                anomalousX=True,
                normalize=True,
            )
        )
        pcds = pcds.isel(pcrank=slice(get_num_pcs(pcds.vrnce, 0.999)))
        odir = Path(__file__).parent.absolute() / "MERRA2/stats"
        pcds.to_netcdf(odir / f"MERRA2.{v}_vint_EOFs_all_days.1980_2023.nc")
