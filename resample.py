import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import xarray as xr
import pandas as pd
from tqdm import tqdm
import time
from inspect import stack

# from dask.diagnostics import ProgressBar

# root data directory for resampling
# DATA_ROOT = os.environ["project"] + "/ERA5/"
DATA_ROOT = "/project/nnn/samuelsmith/lwa_precursors/MERRA2/"
# DATA_ROOT = os.environ["project"] + "/OAFLUX/regridded/"
# DATA_ROOT = "/N/project/pfec_climo/reanalysis/MERRA2"
# variable to resample
VAR = "TLML"
# subset of variables within a file to resample. None for all
SUBVARS = None
# infreq is name of input directory frequency.
# infreq also parsed to determine expected number of files. see note below
INFREQ = "hrly"
# ofreq is name of output directory frequency, but also is parsed to
# determine the resampling frequency. see note below
OFREQ = "6hrly"
# whether output files should combine input files to monthly, bool
COMBINE_TO_MONTHLY = True
# files are assumed to contain one full day/ one full month
DAILY_FILEFREQ = True
# whether or not to delete files after resampling, bool
DELETE_AFTER_RESAMPLE = False
# whether files should have a timestamp displaying the time range contained within, bool
RNG_TIMESTAMP = False

# if DRY_RUN is true, output info about input files found and what the output will look
# like without performing any calculations or making any dirs/files
DRY_RUN = False
# TEST_PARALLEL allows a trial run with a single file (does compute and create a file
# and dirs) in serial to estimate resources and ensure no errors
# if TEST_PARALLEL is False, resampling will be done in parallel with resources
# as specified below. Any corrupt files will be skipped with a warning
TEST_PARALLEL = False
NCORES = 6

if DRY_RUN:  # only works in serial
    TEST_PARALLEL = True

# assume directory tree with structure: root/frequency/variable/(subdirs)/files
# subdirs optional, copied from input directory
INDIR = Path(f"{DATA_ROOT}/{INFREQ}/{VAR}/")
ODIR = Path(f"{DATA_ROOT}/{OFREQ}/{VAR}/")
# INDIR = Path(f"{DATA_ROOT}/")
# ODIR = Path(f"{os.environ['merra2']}/{OFREQ}/{VAR}/")


# return the position of the first alphabet character in a word
def first_alpha(word):
    return [i for i, c in enumerate(word) if c.isalpha()][0]


# this function parses frequency strings to be used by the rest of the code
def parse_freq(freqstr: str) -> Tuple[str, int]:
    """
    all frequency strings above should be integer multiple of hourly, daily, monthly
    multiples should have a number directly preceding the interval
    (interval shorthands are acceptable, 1 is implied if no number given)
    ex. 6hrly/monthly/5daily/3dly/6mnthly/3Hourly/1daily
    NOT 6-hourly/1 monthly/twicedaily/4Xdaily/3sec

    returns pandas compatiable code and integer for resampling
    """

    # get first non-numeric character
    if freqstr == "seasonal":
        freq_code = "Q-FEB"
    else:
        freq_code = freqstr[first_alpha(freqstr)].upper()
    # get everything preceding (or assume 1)
    if freqstr[0].isdigit():
        freq_mult = int(freqstr[slice(None, first_alpha(freqstr))])
    else:
        freq_mult = 1
    return freq_code, freq_mult


INFREQ_CODE, INFREQ_MULT = parse_freq(INFREQ)
OFREQ_CODE, OFREQ_MULT = parse_freq(OFREQ)
ALLOWED_OFREQS = ["H", "D", "M", "Q-FEB"]
ALLOWED_INFREQS = ["H", "D"]


def update_hist(attrs_dict: Dict, desc: str):
    """
    update a netcdf file history with a description of changes, timestamp, and file name

    Parameters
    ----------
    attrs_dict : Dict
        an attribute dictionary for a dataset
    desc : str
        a description of changes made to the data
    """
    attrs_dict.update(
        {
            "history": (
                attrs_dict.get("history", "")
                + f'; {time.strftime("%B %d %Y %H%MZ",time.gmtime())} '
                f"- {desc} by {stack()[1].filename}"
            ).lstrip("; ")
        }
    )
    return


def find_all_files(file: Path) -> List[Path]:
    """
    get all files needed for resampling based off of the single input file

    Parameters
    ----------
    file : Path
        first file in potential range of files, files assumed to
        have minimum one day

    Returns
    -------
    list of Path objects
        all files needed for resampling
    """
    if COMBINE_TO_MONTHLY:
        timestr = file.name.split(".")[-2]
        in_tstr = timestr.split("_")[0]
        tobj = pd.to_datetime(
            in_tstr, format="%Y%m%d%H"[: (len(in_tstr) - 4 + 2 * DAILY_FILEFREQ + 2*RNG_TIMESTAMP)]
        )
        ntimes = length_of_period(
            pd.Period(tobj, freq="M" if OFREQ_CODE != "Q-FEB" else "Q-FEB")
        )
        time_list = pd.date_range(
            tobj, freq="D" if DAILY_FILEFREQ else "M", periods=ntimes
        )
        if RNG_TIMESTAMP:
            start_list = time_list.to_period().start_time.strftime(r"%Y%m%d")
            end_list = time_list.to_period().end_time.strftime(r"%Y%m%d")
            if INFREQ_CODE == "H":
                timestr_list = [
                    f"{st}00_{et}{timestr[-2:]}" for st, et in zip(start_list, end_list)
                ]
            else:
                timestr_list = [f"{st}_{et}" for st, et in zip(start_list, end_list)]
        else:
            timestr_list = time_list.strftime(r"%Y%m%d"[: 4 + 2 * DAILY_FILEFREQ])
        file_list = [
            INDIR
            / str(file.relative_to(INDIR))
            .replace(timestr, tstr)
            .replace(timestr[:4] + "/", tstr[:4] + "/")
            for tstr in timestr_list
        ]
    else:
        file_list = [file]
    return file_list


def length_of_period(period: pd.Period) -> int:
    """get number of days/months in a pandas Period"""
    if DAILY_FILEFREQ:
        return (period.end_time - period.start_time).round("D").days
    else:
        return 1 if OFREQ_CODE[0] != "Q" else 3


def rename_for_resample(file: Path) -> Path:
    """
    map the single input file name (first in range) to the single output file
    There is not currently support for multiple output files

    Parameters
    ----------
    file : Path
        first file in range

    Returns
    -------
    Path
        output file name
    """

    in_timestr = file.name.split(".")[-2]
    # adjust timestamp to match new freq
    if (OFREQ_CODE == INFREQ_CODE) or (OFREQ_CODE == "Q-FEB"):
        extralen = None
    else:
        extralen = -2 * (
            ALLOWED_OFREQS.index(OFREQ_CODE) - ALLOWED_INFREQS.index(INFREQ_CODE)
        )
    if RNG_TIMESTAMP:
        start_tstr, end_tstr = in_timestr.split("_")
        if COMBINE_TO_MONTHLY:
            end_tstr = (
                pd.to_datetime(start_tstr, format="%Y%m%d%H")
                + pd.DateOffset(
                    months=1 if OFREQ_CODE != "Q-FEB" else 3,
                    days=-1 if OFREQ_CODE != "D" else -OFREQ_MULT,
                )
            ).strftime("%Y%m%d00")[: len(start_tstr)]

        if OFREQ_CODE == "H":
            end_tstr = end_tstr[:-2] + f"{24-OFREQ_MULT}"
        elif OFREQ_CODE == "Q-FEB":
            extralen = -2 * (
                ALLOWED_OFREQS.index("M") - ALLOWED_INFREQS.index(INFREQ_CODE)
            )
        out_timestr = start_tstr[:extralen] + "_" + end_tstr[:extralen]
        outfile = ODIR / str(file.relative_to(INDIR)).replace(in_timestr, out_timestr)
    else:
        subdirs = str(file.relative_to(INDIR).parent)
        fname = file.name
        # fname = f"MERRA2.inst{INFREQ_MULT}_3d_ana_Np.{VAR}S." + file.name
        in_desc = fname.split(".")[1]
        pref = in_desc.split("_")[0]
        out_desc = in_desc.replace(
            pref,
            pref[:-1] + str(OFREQ_MULT)
            if OFREQ_CODE == "H"
            else pref[:-1] + OFREQ_CODE[0],
        )
        if COMBINE_TO_MONTHLY:
            out_timestr = in_timestr[:extralen]
        else:
            out_timestr = in_timestr
        outfile = (ODIR / subdirs) / fname.replace(in_timestr, out_timestr).replace(
            in_desc, out_desc
        ).rstrip("34")

    return outfile


def resample_and_combine(f: Path, susbset_vars: Optional[List[str]] = SUBVARS):
    """
    Using a single input file, resample to the specified output frequency, finding
    other files in the resampling range as necessary (if COMBINE_TO_MONTHLY is True)

    output file will contain either a month's worth of data at the new frequency
    or a single sample at the output frequency

    Parameters
    ----------
    f : Path
        first input file of range of files needed for resampling, assumed to contain
        a minimum of one day

    subset_vars: List of strings, optional
        list of variables within a larger dataset to extract, otherwise all
    """
    fname = f.name
    logging.info(f"beginning {VAR} day " + fname.split(".")[-2][:8])
    infile_list = sorted(find_all_files(f))
    outfile = rename_for_resample(f)

    try:
        ds0 = xr.open_dataset(f, decode_times=False)
        if susbset_vars is None:
            susbset_vars = list(ds0.data_vars)
        if COMBINE_TO_MONTHLY:
            ds = xr.open_mfdataset(
                infile_list, join="override", concat_dim="time", combine="nested"
            )[
                susbset_vars
            ]  # .sel(lev=950, method="nearest")
        else:
            ds = ds0[susbset_vars]
    except Exception as e:
        if TEST_PARALLEL:
            raise e
        else:
            logging.warning(f"{fname} failed with error {e}")
            return

    if OFREQ_CODE == "Q-FEB":
        ofreqstr = "QS-DEC"
    elif OFREQ_CODE == "M":
        ofreqstr = "MS"
    else:
        ofreqstr = OFREQ_CODE
    
    if DRY_RUN:
        outtimes = ds.time.resample(time=str(OFREQ_MULT) + ofreqstr).mean("time")
        try:
            intime_list = list(
                ds.get_index("time").to_datetimeindex().strftime("%Y-%m-%dT%H")
            )
            outtime_list = list(
                outtimes.get_index("time").to_datetimeindex().strftime("%Y-%m-%dT%H")
            )
        except AttributeError:
            intime_list = list(ds.get_index("time").strftime("%Y-%m-%dT%H"))
            outtime_list = list(outtimes.get_index("time").strftime("%Y-%m-%dT%H"))
        logging.info(f"input file: {f}")
        logging.info(f"output file: {outfile}")
        logging.info("variables: " + ", ".join(ds.data_vars))
        logging.info("intimes: " + ", ".join(intime_list))
        logging.info("outtimes: " + ", ".join(outtime_list))
        if DELETE_AFTER_RESAMPLE:
            logging.warning("INPUTS TO BE DELETED AFTER RESAMPLE")
    else:
        # need to make year sub directory
        outfile.parent.mkdir(parents=True, exist_ok=True)

        ds.load()
        ds_res = ds.resample(time=str(OFREQ_MULT) + ofreqstr).mean(
            "time", keep_attrs=True
        )

        # copy/update attrs
        ncattrs = ds0.attrs
        tattrs = {
            att: ds0.time.encoding.get(att, dflt)
            for att, dflt in [
                ("units", "hours since 1970-01-01 00:00:00"),
                ("calendar", "gregorian"),
            ]
        }
        update_hist(ncattrs, "resampled")
        # remove since it slows things down
        if "unlimited_dims" in ds_res.encoding:
            ds_res.encoding.pop("unlimited_dims")
        ds_res.attrs = ncattrs
        ds_res.time.encoding = tattrs
        # ds_res[VAR].attrs["long_name"] = f"{int(ds_res.lev.values)}hPa " + ds_res[
        #     VAR
        # ].attrs.get("long_name", VAR)
        # ds_res = ds_res.rename({VAR: VAR + "S"})
        # with ProgressBar():
        ds_res.to_netcdf(outfile)
        if DELETE_AFTER_RESAMPLE:
            if os.path.exists(outfile):
                for file in infile_list:
                    os.remove(file)
        logging.info(VAR + " " + fname.split(".")[-2][:8] + " written!")
    return


if __name__ == "__main__":
    from multiprocessing.pool import Pool

    if TEST_PARALLEL:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    INSTEM = "nc"

    # input checking
    assert (INFREQ_CODE in ALLOWED_INFREQS) and (
        OFREQ_CODE in ALLOWED_OFREQS
    ), "only hourly/daily/monthly supported"
    if INFREQ_CODE == OFREQ_CODE:
        assert OFREQ_MULT >= INFREQ_MULT, "code designed only for downsampling"
    else:
        assert ALLOWED_INFREQS.index(INFREQ_CODE) < ALLOWED_OFREQS.index(
            OFREQ_CODE
        ), "code designed only for downsampling"
    if COMBINE_TO_MONTHLY:
        if OFREQ_CODE == "M":
            assert OFREQ_MULT == 1, (
                "if monthly output resampling from daily,"
                " max resampling frequency is 1 day"
            )

    globber = f"*.{INSTEM}"
    # going from daily to monthly needs special globbing to get only first day of month
    if COMBINE_TO_MONTHLY and DAILY_FILEFREQ:
        if RNG_TIMESTAMP:
            globber = f"*.??????0100_??????????.{INSTEM}"
        else:
            globber = f"*.??????01.{INSTEM}"
            # globber = f"??????01.{INSTEM}"

    infiles = list(INDIR.rglob(globber))  # files to be resampled
    # infiles = [f for f in infiles if int(f.name.split(".")[-2][4:6]) in [12, 1, 2]]
    if OFREQ_CODE == "Q-FEB":
        # infiles = [f for f in infiles if f.name.split(".")[-2][4:6] == "12"]
        infiles = [f for f in infiles if int(f.name.split(".")[-2][4:6]) % 3 == 0]
    outfiles = set(ODIR.rglob("*.nc"))  # files already resampled

    outfile2infile = {rename_for_resample(file_obj): file_obj for file_obj in infiles}
    leftfiles = set(outfile2infile) - outfiles
    do_files = sorted([outfile2infile[outfile] for outfile in leftfiles])
    if TEST_PARALLEL:
        logging.info(f"{len(do_files)} remaining")
        for f2do in do_files[:1]:
            resample_and_combine(f2do)
    else:
        # with open(
        #     os.environ["HOME"] + "/resamp_era_" + os.environ["SLURM_JOB_ID"] + ".err",
        #     "w",
        # ) as errf:
        with Pool(NCORES) as mypool:
            list(
                tqdm(
                    mypool.imap_unordered(resample_and_combine, do_files, 1),
                    total=len(do_files),
                    ascii=True,
                )
            )  # ,file=errf))
