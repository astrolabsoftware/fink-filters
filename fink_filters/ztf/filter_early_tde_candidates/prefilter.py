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

import pyspark.sql.functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import IntegerType, DoubleType

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord

from fink_utils.spark.utils import concat_col
from fink_filters.ztf.classification import extract_fink_classification


@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
def nneg(isdiffpos: pd.Series) -> pd.Series:
    """Number of negative detections"""
    nneg = isdiffpos.apply(lambda x: np.sum(np.array(x) == "f"))
    return nneg


@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
def nfid_1(fid: pd.Series, mag: pd.Series) -> pd.Series:
    """Number of fid==1 detections"""
    num = []
    for i, fid1 in enumerate(fid):
        num.append(np.sum((np.array(fid1) == 1) & np.isfinite(mag[i])))

    return pd.Series(num)


@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
def nfid_2(fid: pd.Series, mag: pd.Series) -> pd.Series:
    """Number of fid==2 detections"""
    num = []
    for i, fid1 in enumerate(fid):
        num.append(np.sum((np.array(fid1) == 2) & np.isfinite(mag[i])))

    return pd.Series(num)


@pandas_udf(DoubleType(), PandasUDFType.SCALAR)
def calculate_galactic_b(ra: pd.Series, dec: pd.Series) -> pd.Series:
    """Compute galactic b from (ra, dec)."""
    if len(ra) == 0:
        return pd.Series(np.zeros_like(ra))  # or any default values

    gal_b = SkyCoord(
        ra.to_numpy(), dec.to_numpy(), frame="icrs", unit="deg"
    ).galactic.b.value

    return pd.Series(gal_b)


def get_slope(x, y, dy):
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    dy = np.atleast_1d(dy)

    idx = np.isfinite(x) & np.isfinite(y) & np.isfinite(dy)

    if len(x[idx]) < 3 or np.ptp(x[idx]) == 0:
        return 0, 0

    p, cov = np.polyfit(
        x[idx] - np.mean(x[idx]), y[idx] - np.mean(y), 1, w=1 / dy[idx], cov="unscaled"
    )
    dp = np.sqrt(np.diag(cov))

    return p[0], dp[0]


def is_rising(
    jd,
    flux,
    fluxerr,
    fid,
    upper=None,
    nsigmas_rise=2,
    nsigmas_decay=1,
    nsigmas_slope=3,
    use_slope=True,
    verbose=False,
):
    idx0 = flux == flux

    is_decay = False
    is_rise = False

    for fid1 in [1, 2]:
        idx = idx0 & (fid == fid1)
        uidx = ~idx0 & (fid == fid1)
        if np.sum(idx) < 2:
            continue

        # Last point is significantly lower than the max?..
        diff = flux[idx][:-1] - flux[idx][-1]
        differr = np.hypot(fluxerr[idx][:-1], fluxerr[idx][-1])

        if np.any(diff > nsigmas_decay * differr):
            is_decay = True

        # Last point is significantly higher than the min?..
        diff = flux[idx][-1] - flux[idx][:-1]

        if np.any(diff > nsigmas_rise * differr):
            is_rise = True

        if np.sum(idx) >= 3 and use_slope:
            # Slope is significantly positive?..
            slope, serr = get_slope(jd[idx], flux[idx], fluxerr[idx])
            if verbose:
                print("slope", slope, "+/-", serr)
            if slope > nsigmas_slope * serr:
                is_rise = True

        # Any point is significantly lower than the previous?..
        diff = flux[idx][:-1] - flux[idx][1:]
        differr = np.hypot(fluxerr[idx][:-1], fluxerr[idx][1:])

        if np.any(diff > nsigmas_decay * differr):
            is_decay = True

        # Last point is significantly higher than any (prior) upper limit?..
        if upper is not None and np.sum(uidx) > 0:
            diff = flux[idx][-1] - upper[uidx]
            differr = fluxerr[idx][-1]

            if np.any(diff > nsigmas_rise * differr):
                is_rise = True

    if verbose:
        print("rise", is_rise, "decay", is_decay)

    return is_rise and not is_decay


# TODO: merge into fink_utils.photometry.mag2fluxcal_snana
def mag2fluxcal(magpsf, sigmapsf, isdiffpos=None):
    """Conversion from magnitude to FLUXCAL from SNANA manual"""
    if magpsf is None:
        return None, None

    fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
    fluxcal_err = 9.21034 * 10**10 * np.exp(-0.921034 * magpsf) * sigmapsf

    if isdiffpos is not None:
        sign = np.where(isdiffpos == "f", -1, 1)
    else:
        sign = 1

    return sign * fluxcal, fluxcal_err


@pandas_udf(IntegerType(), PandasUDFType.SCALAR)
def is_rising_u(
    jd: pd.Series,
    mag: pd.Series,
    magerr: pd.Series,
    isdiffpos: pd.Series,
    fid: pd.Series,
) -> pd.Series:
    result = []

    for jd1, mag1, magerr1, isdiffpos1, fid1 in zip(jd, mag, magerr, isdiffpos, fid):
        flux1, fluxerr1 = mag2fluxcal(mag1, magerr1, isdiffpos1)

        result.append(
            is_rising(jd1, flux1, fluxerr1, fid1, nsigmas_rise=2, nsigmas_decay=1)
        )

    return pd.Series(result)


def prefilter_alerts(df):
    """Filter alerts prior to lightcurve level processing"""
    # Remove MPC
    df = df.filter(df["roid"] != 3)

    # filter for wanted Simbad classes
    wanted = [
        "",
        "X",
        "IR",
        "Radio",
        "MIR",
        "NIR",
        "HH",
        "HI",
        "HII",
        "HighPM*",
        "LensedImage",
        "LensingEv",
        "Maser",
        "MolCld",
        "PartofCloud",
        "Radio(sub-mm)",
        "Blue",
        "Possible_lensImage",
        "Unknown",
        "Radio(mm)",
        "denseCore",
        "Radio(cm)",
        "UV",
        "PN",
        "PN?",
        "EmObj",
        "DkNeb",
        "Transient",
        "Candidate_LensSystem",
        "FIR",
        "multiple_object",
        "GravLensSystem",
        "Bubble",
        "Cloud",
        "SFregion",
        "Inexistent",
        "gamma",
        "GravLens",
        "HVCld",
        "Candidate_Lens",
        "ISM",
        "Void",
        "RfNeb",
        "HIshell",
        "Outflow",
        "radioBurst",
        "Region",
        "Globule",
        "outflow?",
        "ComGlob",
        "GinCl",
        "Galaxy",
        "AGN",
        "GiC",
        "Sy1",
        "Sy2",
        "AGN_Candidate",
        "QSO",
        "Seyfert_1",
        "Seyfert_2",
        "LINER",
        "EmG",
        "RadioG",
        "BClG",
        "LSB_G",
        "LensedG",
        "LensedQ",
        "GroupG",
        "PartOfG",
        "BLLac",
        "GinPair",
        "Possible_ClG",
        "Possible_G",
        "Possible_GrG",
        "GinGroup",
        "HII_G",
        "Blazar",
        "ClG",
        "QSO_Candidate",
        "Seyfert",
        "Blazar_Candidate",
        "StarburstG",
        "IG",
        "SuperClG",
        "PartofG",
        "Compact_Gr_G",
        "PairG",
        "BLLac_Candidate",
        "BlueCompG",
        "Seyfert2",
        "Seyfert1",
    ]

    df = df.filter(df["cdsxmatch"].isin(wanted))

    # Minimal number of points in the alert
    df = df.filter(df["nalerthist"] >= 5)

    # Vectorize historical measurements
    cols_to_concatenate = ["jd", "fid", "magpsf", "diffmaglim", "sigmapsf", "isdiffpos"]
    for _ in cols_to_concatenate:
        df = concat_col(df, _, prefix="c")

    cols = (
        [
            # Useful pre-existing columns
            "objectId",
            "candidate.ra",
            "candidate.dec",
            "candidate.candid",
            "candidate.distnr",
            "candidate.distpsnr1",
            "candidate.magnr",
            "candidate.sigmagnr",
            "candidate.nbad",
            "nalerthist",
        ]
        + ["c" + _ for _ in cols_to_concatenate]
        + [
            # For classification
            "cdsxmatch",
            "roid",
            "mulens",
            "snn_snia_vs_nonia",
            "snn_sn_vs_all",
            "rf_snia_vs_nonia",
            "candidate.ndethist",
            "candidate.drb",
            "candidate.classtar",
            "candidate.jd",
            "candidate.jdstarthist",
            "rf_kn_vs_nonkn",
            "tracklet",
        ]
    )

    df = df.select(cols)

    # No more than 1 negative points in the alert
    df = df.withColumn("nneg", nneg(df["cisdiffpos"]))
    df = df.filter(df["nneg"] <= 1)
    df = df.drop("nneg")

    # At least one point in every filter in the alert
    df = df.withColumn("nfid_1", nfid_1(df["cfid"], df["cmagpsf"]))
    df = df.filter(df["nfid_1"] > 0)
    df = df.drop("nfid_1")

    df = df.withColumn("nfid_2", nfid_2(df["cfid"], df["cmagpsf"]))
    df = df.filter(df["nfid_2"] > 0)
    df = df.drop("nfid_2")

    # Above Galactic plane: |b|>=20
    df = df.withColumn("b", calculate_galactic_b("ra", "dec"))
    df = df.filter(F.abs(F.col("b")) >= 20)
    df = df.drop("b")

    # Filter out decaying lightcurves
    df = df.withColumn(
        "is_rising",
        is_rising_u(
            df["cjd"], df["cmagpsf"], df["csigmapsf"], df["cisdiffpos"], df["cfid"]
        ),
    )
    df = df.filter(df["is_rising"] == 1)
    df = df.drop("is_rising")

    df = df.withColumn(
        "classification",
        extract_fink_classification(
            df["cdsxmatch"],
            df["roid"],
            df["mulens"],
            df["snn_snia_vs_nonia"],
            df["snn_sn_vs_all"],
            df["rf_snia_vs_nonia"],
            df["ndethist"],
            df["drb"],
            df["classtar"],
            df["jd"],
            df["jdstarthist"],
            df["rf_kn_vs_nonkn"],
            df["tracklet"],
        ),
    )

    return df
