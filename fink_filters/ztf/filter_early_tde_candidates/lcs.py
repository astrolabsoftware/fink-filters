# Copyright 2019-2025 AstroLab Software
# Author: Sergey Karpov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from astropy.coordinates import SkyCoord

import os
import io
import requests
import pandas as pd

from dustmaps.config import config as dustconfig
import dustmaps.sfd

from light_curve.light_curve_py import RainbowFit

import matplotlib.pyplot as plt

import logging

from fink_filters.ztf.filter_early_tde_candidates.prefilter import mag2fluxcal

dustconfig["data_dir"] = "/tmp"

COLORS_ZTF = {1: "#15284F", 2: "#F5622E"}

API_ENDPOINT = "https://api.fink-portal.org/api/v1/objects"

# Filters ZTF
filt_conv = {
    1: "g",
    2: "r",
    3: "i",
}  # Conversion between filter ID (int) and filter name (str)
band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0}  # Bands in angstroms for rainbow

_LOG = logging.Logger("Early TDE")


# TNS
tns = None  # Global cache for TNS entries


def get_tns_info(oid=None, ra=None, dec=None, sr=5 / 3600, return_types=False):
    """Return TNS classification for the object, if any"""
    global tns

    if tns is None:
        r = requests.post(
            "https://api.fink-portal.org/api/v1/resolver",
            json={"resolver": "tns", "name": "", "nmax": 1000000},
        )

        tns = pd.read_json(io.BytesIO(r.content))

    if oid is not None:
        # Search by ZTF objectId
        idx = tns["d:internalname"] == oid
    elif ra is not None and dec is not None:
        # Search by coordinates
        idx = (
            SkyCoord(tns["d:ra"].values, tns["d:declination"].values, unit="deg")
            .separation(SkyCoord(ra, dec, unit="deg"))
            .deg
            < sr
        )
    else:
        return None

    if return_types:
        idx &= tns["d:type"] != "nan"
        return list(tns[idx]["d:type"])

    result = []

    for _, row in tns[idx].iterrows():
        res = row["d:fullname"]
        if row["d:type"] and row["d:type"] != "nan":
            res += " (" + row["d:type"]
            if row["d:redshift"] > 0:
                res += f" z={row['d:redshift']}"
            res += ")"

        if res not in result:
            result.append(res)

    return " | ".join(result)


# Extinction
sfd = None
# Coefficients for ZTF filters for G23 extinction law from dust_extinction for Rv=3.1
Av = {1: 3.681, 2: 2.635, 3: 1.944}


def prepare_sfd_data():
    path = dustmaps.sfd.data_dir()
    path = os.path.join(path, "sfd")

    if not os.path.exists(path):
        _LOG.warning("No SFD data for dustmaps, downloading it")
        dustmaps.sfd.fetch()


def deredden(flux, fid, ra, dec):
    """De-reddening using SFD from dustmaps"""
    global sfd

    if sfd is None:
        prepare_sfd_data()
        sfd = dustmaps.sfd.SFDQuery()

    Ebv = sfd(SkyCoord(ra, dec, unit="deg"))

    return flux / 10 ** (-0.4 * (Ebv * Av[fid]))


def deredden_pdf(pdf, ra=None, dec=None):
    """De-reddening the dataframe in-place using SFD from dustmaps"""
    global sfd

    if sfd is None:
        prepare_sfd_data()
        sfd = dustmaps.sfd.SFDQuery()

    Ebv = sfd(SkyCoord(ra, dec, unit="deg"))
    for fid in [1, 2, 3]:
        idx = pdf["i:fid"] == fid
        pdf.loc[idx, "FLUXCAL"] /= 10 ** (-0.4 * (Ebv * Av[fid]))
        pdf.loc[idx, "FLUXCALERR"] /= 10 ** (-0.4 * (Ebv * Av[fid]))

        if "FLUXCALUPPER" in pdf:
            pdf.loc[idx, "FLUXCALUPPER"] /= 10 ** (-0.4 * (Ebv * Av[fid]))


# Light curves
def request_lc(oid):
    r = requests.post(
        "https://api.fink-portal.org/api/v1/objects",
        json={
            "objectId": oid,
            "output-format": "json",
            "withupperlim": "True",
            "columns": "d:tag, d:nalerthist, i:ndethist, i:jd, i:magpsf, i:sigmapsf, i:isdiffpos, i:diffmaglim, i:fid, i:distnr, i:magnr, i:ra, i:dec",
        },
    )

    if r.status_code != 200:
        _LOG.warning(
            "Error getting archival light curve for {}: {}".format(oid, r.content)
        )
        return None

    pdf = pd.read_json(io.BytesIO(r.content))
    pdf = pdf.sort_values("i:jd", inplace=False)

    # Remove other bands?..
    pdf = pdf[(pdf["i:fid"] == 1) | (pdf["i:fid"] == 2)]

    pdf["FLUXCAL"], pdf["FLUXCALERR"] = mag2fluxcal(
        pdf["i:magpsf"], pdf["i:sigmapsf"], pdf["i:isdiffpos"]
    )
    pdf["FLUXCALUPPER"] = 10 ** (11 - 0.4 * pdf["i:diffmaglim"])

    ra = np.nanmean(pdf["i:ra"])
    dec = np.nanmean(pdf["i:dec"])

    deredden_pdf(pdf, ra, dec)

    return pdf


def request_lc_snad(ra, dec, sr_arcsec=1.5):
    r = requests.get(
        "https://db.ztf.snad.space/api/v3/data/latest/circle/full/json",
        params={"ra": ra, "dec": dec, "radius_arcsec": sr_arcsec},
    )

    if r.status_code != 200:
        _LOG.warning(
            "Error getting SNAD light curve for {} {} {}: {}".format(
                ra, dec, sr_arcsec, r.content
            )
        )
        return None

    lc = []
    for v in r.json().values():
        lc1 = pd.DataFrame(v["lc"])
        lc1["filter"] = v["meta"]["filter"]
        lc1["i:fid"] = {"zg": 1, "zr": 2, "zi": 3}.get(v["meta"]["filter"])
        lc.append(lc1)

    if len(lc):
        pdf = pd.concat(lc, ignore_index=True)

        pdf["i:jd"] = pdf["mjd"] + 2400000.5
        pdf["FLUXCAL"], pdf["FLUXCALERR"] = mag2fluxcal(pdf["mag"], pdf["magerr"])

        deredden_pdf(pdf, ra, dec)
    else:
        pdf = None

    return pdf


def get_lc(row, prefix="i:"):
    # Extract light curve from the data row
    pdf = pd.DataFrame({
        "i:" + _: row[prefix + _]
        for _ in ["fid", "jd", "magpsf", "sigmapsf", "isdiffpos"]
    })

    pdf["FLUXCAL"], pdf["FLUXCALERR"] = mag2fluxcal(
        pdf["i:magpsf"], pdf["i:sigmapsf"], pdf["i:isdiffpos"]
    )

    if prefix + "diffmaglim" in row:
        pdf["i:diffmaglim"] = row[prefix + "diffmaglim"]
        pdf["FLUXCALUPPER"] = 10 ** (11 - 0.4 * pdf["i:diffmaglim"])

    pdf = pdf.sort_values("i:jd", inplace=False, ignore_index=True)

    # Remove other bands?..
    pdf = pdf[(pdf["i:fid"] == 1) | (pdf["i:fid"] == 2)]

    deredden_pdf(pdf, row["ra"], row["dec"])

    return pdf


def plot_lc(pdf, show_zero=False, show_limits=False, ms=None):
    for fid, cn, c in [(1, "g", COLORS_ZTF[1]), (2, "r", COLORS_ZTF[2])]:
        idx = pdf["i:fid"] == fid
        mjd = pdf["i:jd"] - 2400000.5

        plt.errorbar(
            mjd[idx],
            pdf["FLUXCAL"][idx],
            pdf["FLUXCALERR"][idx],
            fmt=".",
            c=c,
            alpha=0.2,
        )
        plt.plot(mjd[idx], pdf["FLUXCAL"][idx], ".", color=c, alpha=1, ms=ms, label=cn)

        if show_limits and "FLUXCALUPPER" in pdf:
            idx &= ~np.isfinite(pdf["FLUXCAL"])
            plt.plot(mjd[idx], pdf["FLUXCALUPPER"][idx], "^", alpha=0.3, color=c)

    plt.grid(alpha=0.2)
    if show_zero:
        plt.axhline(0, ls="--", color="black", alpha=0.2)


# Features
feature = None


def extract_features(sub, nsamples=None):
    global feature

    if feature is None:
        feature = RainbowFit.from_angstrom(
            band_wave_aa,
            with_baseline=False,
            temperature="constant",
            bolometric="sigmoid",
        )

    jd = sub["i:jd"].to_numpy()
    flux, fluxerr = sub["FLUXCAL"].to_numpy(), sub["FLUXCALERR"].to_numpy()
    flux_upper = sub["FLUXCALUPPER"].to_numpy()
    fluxerr_upper = sub["FLUXCALUPPER"].to_numpy() / 5

    idx_good = flux == flux

    band = np.array([filt_conv[_] for _ in sub["i:fid"]])

    try:
        params, errors, cov = feature._eval_and_get_errors(
            t=jd,
            m=np.where(idx_good, flux, flux_upper),
            sigma=np.where(idx_good, fluxerr, fluxerr_upper),
            band=band,
            upper_mask=~idx_good,
            return_covariance=True,
            debug=True,
        )

        result = {}
        result["params"] = params
        result["errors"] = errors
        result["cov"] = cov

        if nsamples is not None:
            samples = []

            while True:
                params1 = np.random.multivariate_normal(params[:-1], cov)
                # Sanity checks
                if params1[feature.p["amplitude"]] < 0:
                    continue

                samples.append(params1)
                if len(samples) >= nsamples:
                    result["samples"] = samples
                    break

        return result
    except KeyboardInterrupt:
        raise
    except Exception:  # FIXME
        return None


def print_features(res1):
    global feature

    for i, name in enumerate(feature.names):
        value, error = res1["params"][i], res1["errors"][i]

        if name in ["rise_time", "T", "amplitude"]:
            snr = f" (S/N = {value / error:.1f})"
        else:
            snr = ""

        if name == "reference_time":
            value -= 2400000.5

        if name == "T":
            name = "temperature"

        print(f"{name} = {value:.1f} +/- {error:.1f}{snr}")
    print(f"r_chisq = {res1['params'][-1]:.1f}")


def plot_features(
    sub, res1, nsamples=100, prior=100, jd_min=None, jd_max=None, extra=None, **kwargs
):
    global feature

    if jd_min is not None:
        sub = sub[sub["i:jd"] >= jd_min]
    if jd_max is not None:
        sub = sub[sub["i:jd"] <= jd_max]

    plot_lc(sub, **kwargs)

    if prior:
        t = np.linspace(np.max(sub["i:jd"]) - prior, np.max(sub["i:jd"]) + 1, 1000)
    else:
        t = np.linspace(np.min(sub["i:jd"]) - 1, np.max(sub["i:jd"]) + 1, 1000)

    for _ in range(nsamples):
        params1 = np.random.multivariate_normal(res1["params"][:-1], res1["cov"])
        if (
            params1[feature.p["amplitude"]] < 0
            or params1[feature.p["rise_time"]] < 0
            or params1[feature.p["T"]] < 0
        ):
            continue
        for _, c, col in ((0, "g", COLORS_ZTF[1]), (1, "r", COLORS_ZTF[2])):
            plt.plot(
                t - 2400000.5, feature.model(t, c, *params1), color=col, alpha=0.03
            )

    for c, col in [("g", COLORS_ZTF[1]), ("r", COLORS_ZTF[2])]:
        plt.plot(
            t - 2400000.5,
            feature.model(t, [c], *res1["params"]),
            "--",
            color=col,
            alpha=0.5,
        )

    texts = []
    for i, name in enumerate(feature.names):
        value, error = res1["params"][i], res1["errors"][i]

        if name in ["rise_time", "T", "amplitude"]:
            snr = f" (S/N = {value / error:.1f})"
        else:
            snr = ""

        if name == "reference_time":
            value -= 2400000.5

        if name == "T":
            name = "temperature"

        texts.append(f"{name} = {value:.1f} +/- {error:.1f}{snr}")
    texts.append(f"r_chisq = {res1['params'][-1]:.1f}")

    if extra is not None:
        texts += extra  # TODO: check whether it is a list

    plt.text(
        0.05,
        0.95,
        "\n".join(texts),
        transform=plt.gca().transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7},
    )


def cleanup_limits(sub, ignore=False):
    if ignore:
        # Ignore all upper limits
        sub = sub[np.isfinite(sub["FLUXCAL"])]

    elif "FLUXCALUPPER" in sub:
        mask = np.isfinite(sub["i:jd"])

        for fid in [1, 2]:
            idx1 = sub["i:fid"] == fid  # Same filter
            idx2 = np.isfinite(sub["FLUXCAL"])  # Detections

            # jd1 = np.min(sub['i:jd'][idx1 & idx2]) # First detection, same filter
            jd1 = np.min(sub["i:jd"][idx2])  # First detection, any filter
            # flux1 = (sub["FLUXCAL"][sub["i:jd"] == jd1]).values[0]
            idx3 = sub["i:jd"] >= jd1 - 1  # Later than first detection
            # idx3 |= sub['FLUXCALUPPER'] > flux1 # ..or above first detection?..
            idx3 |= sub["FLUXCALUPPER"] > np.nanmin(
                sub["FLUXCAL"][idx1]
            )  # ..or above min flux?..

            mask[idx1 & ~idx2 & idx3] = False

        sub = sub[mask]

    else:
        # Dataset without upper limits defined
        sub = sub.assign(FLUXCALUPPER=np.nan)

    return sub
