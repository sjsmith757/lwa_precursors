import numpy as np
import xarray as xr
from typing import Literal, Optional, Hashable, TypeVar, Container
import warnings

Xobj = TypeVar("Xobj", xr.Dataset, xr.DataArray)


def add_adj(
    da: Xobj, dim: Hashable, edge_mode: str = "drop", mode: str = "right"
) -> Xobj:
    """
    add adjacent elements in an xarray object, returning an object with coordinates
    corresponding to mode

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        the xarray object to add adjacent elements
    dim : Hashable
        dimension to use for adding adjacent elements
    edge_mode : str, optional
        how to handle array boundaries, by default "drop"
        Must be either "drop", "roll", or "fill"
        "drop": remove data at the boundary(s) used by mode
                (a[1] + a[0] first, or a[-2]+a[-1] last)
        "roll": treat boundary as periodic and repeat data across it
                (a[-1] + a[0])
        "fill": repeat data at the boundary (most useful for center mode)
                (a[0]+a[0], a[-1]+a[-1], or a[0]+a[1] and a[-2]+a[-1])
    mode : str, optional
        determines which adjacent elements are added, by default "right"
        Must be either "right", "left", or "center"
        "right":  add the current element with the following element, using the
                  label for the following element (e.g., a[i] + a[i+1])
        "left":   add the current element with the previous element, using the
                  label for the current element (e.g., a[i-1] + a[i])
        "center": add the previous element with the element following the current one,
                  using the label for the current element (e.g., a[i-1] + a[i+1])

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        array with added elements, of dimension dim.size - 1 (for left or right)
        or dim.size - 2 in dimension "dim", or of same size if using roll or fill
    """

    rshift = 1 if (mode == "right") or (mode == "center") else 0
    lshift = -1 if (mode == "left") or (mode == "center") else 0
    if rshift == lshift:
        raise ValueError("mode must be either right, left, or center")

    if edge_mode == "roll":
        rda = da.roll({dim: rshift}, roll_coords=False)
        lda = da.roll({dim: lshift}, roll_coords=False)
    else:
        rda = da.shift({dim: rshift})
        lda = da.shift({dim: lshift})
        if edge_mode == "drop":
            rda = rda.isel({dim: slice(rshift, None)})
            lda = lda.isel({dim: slice(0, -1 if lshift else None)})
        elif edge_mode == "fill":
            rda = rda.fillna(da.isel({dim: 0}) if rshift else np.nan)
            lda = lda.fillna(da.isel({dim: -1}) if lshift else np.nan)
        elif edge_mode == "stop":
            raise ValueError("Where's the fire?")
        else:
            raise ValueError("mode must be either drop, roll, or fill")

    return rda + lda


def mid_interp(
    regrid: xr.DataArray,
    dim: Optional[Hashable] = None,
    axis: Optional[int] = None,
    weight: Optional[xr.DataArray] = None,
    roll: bool = False,
    *,
    _coord: bool = False,
) -> xr.DataArray:
    """
    Midpoint interpolation function allowing for a weighting function
    regrid :: array to be interpolated. Required
    axis :: axis to be interpolated along. Required
    weight :: performs a weighted average while interpolating along the
              specified axis. Default is no weighting.
              Weight should be one dimensional
    roll :: for interpolating between grid points which wrap around
            (i.e. longitude), default is not wrapped

    _coord is for internal use by the function and is not intended for general users. It
    specifies that the input is a coordinate dimension who should be handled differently.
    It is used to update the coordinates on the input data

    :: returns ::
    regrid_mid :: interpolated grid to midpoints of original array
    """

    # first create indices, using tuples to allow for any dimension of array
    if dim is not None:
        axis = regrid.dims.index(dim)
    elif axis is not None:
        dim = regrid.dims[axis]
    else:
        raise ValueError("One of either dim or axis must be provided")

    # coords shouldn't be weighted
    if weight is None or _coord:
        weight = xr.ones_like(regrid[dim], dtype=regrid.dtype)
    else:
        # set code to exit if weight is not one-dimensional
        if weight.size != regrid[dim].size:
            raise ValueError(
                f"weight size {weight.size} should match "
                f"dimension size {regrid[dim].size}"
            )
        weight = weight.squeeze()

    if regrid.chunks is not None:
        orig_chunks = regrid.data.chunksize[axis]
    assert weight is not None  # for mypy
    # divide into routines based off whether we are rolling across the boundary
    # weighted interpolation is equivalent to
    # (point_1*weight_1 + point_2*weight_2)/(weight_1 + weight_2)
    # so if weight_1 = weight_2 it is equivalent to no weight
    weight_mid = add_adj(
        weight, dim, edge_mode="roll" if roll else "drop", mode="left"  # type: ignore
    )
    regrid_mid = add_adj(
        weight * regrid, dim, edge_mode="roll" if roll else "drop", mode="left"
    )
    if _coord and roll:
        # weight always 1 for coords
        regrid_mid[-1] = 2 * regrid_mid[-2].values - regrid_mid[-3].values
    regrid_mid /= weight_mid
    if not _coord:
        new_coords = mid_interp(regrid[dim], dim=dim, roll=roll, _coord=True)
        regrid_mid = regrid_mid.assign_coords({dim: new_coords.values})

    if regrid_mid.chunks is not None:
        regrid_mid = regrid_mid.chunk({dim: orig_chunks})
    regrid_mid.name = regrid.name
    regrid_mid.attrs = regrid.attrs

    return regrid_mid


def match_grid(
    arr: xr.DataArray, dims: Container[Literal["lat", "lev", "lon"]]
) -> xr.DataArray:
    """
    interpolate an array onto the PV grid: half-lon, half-lat, half-half
    (density-weighted) lev

    Parameters
    ----------
    arr : xr.DataArray
        the array to interpolate
    dims : Container["lat","lon","lev"]
        A collection of dimensions to interpolate; they should be a subset of "lat",
        "lev", and "lon". The extra dimensions will be ignored. Not all dimensions are
        required. Order is unimportant.

        Exs. of valid dims:
        .. code-block::
            >>> match_grid(arr, ["lon","lat","lev"])
            >>> match_grid(arr, ["lat","lev"])
            >>> match_grid(arr, ("lon","lat"))
            >>> match_grid(arr, {"lev"})

    Returns
    -------
    xr.DataArray
        the interpolated array
    """

    # check if there are any strings in dims which aren't supported dims
    if set(dims) - {"lat", "lev", "lon"}:
        warnings.warn(
            "unconventional dimention names will not be interpolated", stacklevel=2
        )
    # make a copy so that we can update sequentially
    flx_arr_interpd = arr.copy()
    for c in arr.coords:
        if c != "time":
            flx_arr_interpd[c] = arr[c].astype(arr.dtype)
    # interpolate along each dimension if present
    if "lev" in dims:
        rho_0 = flx_arr_interpd["lev"] / 1000.0
        flx_arr_mid = mid_interp(flx_arr_interpd, dim="lev", weight=rho_0)
        rho_0mid = mid_interp(rho_0, dim="lev")
        rho_0mid = rho_0mid.assign_coords({"lev": flx_arr_mid.lev.values})
        flx_arr_interpd = mid_interp(flx_arr_mid, dim="lev", weight=rho_0mid)
    if "lat" in dims:
        cos_phi = np.cos(np.pi / 180.0 * flx_arr_interpd["lat"])
        flx_arr_interpd = mid_interp(flx_arr_interpd, dim="lat", weight=cos_phi)
    if "lon" in dims:
        flx_arr_interpd = mid_interp(flx_arr_interpd, dim="lon", roll=True)

    return flx_arr_interpd
