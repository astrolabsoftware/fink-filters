# Copyright 2019-2022 AstroLab Software
# Authors: Julien Peloton, Juliette Vlieghe
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
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import numpy as np
import pandas as pd
import datetime
import requests
import os
import logging
from scipy.optimize import curve_fit

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from astropy.time import Time
from astroquery.sdss import SDSS

from fink_utils.photometry.conversion import dc_mag
from fink_utils.xmatch.simbad import return_list_of_eg_host

from fink_filters.tester import spark_unit_tests

def perform_classification(
        objectId, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jdstarthist, ndethist, cdsxmatch, ra, dec, ssdistnr, cjdc,
        cfidc, cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic,
        cisdiffposc) -> pd.Series:
    """
    Return alerts considered as KN candidates.

    The cuts are based on Andreoni et al. 2021 https://arxiv.org/abs/2104.06352

    If the environment variable KNWEBHOOK is defined and match a webhook url,
    the alerts that pass the filter will be sent to the matching Slack channel.

    Parameters
    ----------
    objectId: Spark DataFrame Column
        Column containing the alert IDs
    rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all: Spark DataFrame Columns
        Columns containing the scores for: 'Early SN Ia',
        'Ia SN vs non-Ia SN', 'SN Ia and Core-Collapse vs non-SN events'
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    jdstarthist: Spark DataFrame Column
        Column containing earliest Julian dates of epoch [days]
    ndethist: Spark DataFrame Column
        Column containing the number of prior detections (theshold of 3 sigma)
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    ra: Spark DataFrame Column
        Column containing the right Ascension of candidate; J2000 [deg]
    dec: Spark DataFrame Column
        Column containing the declination of candidate; J2000 [deg]
    ssdistnr: Spark DataFrame Column
        distance to nearest known solar system object; -999.0 if none [arcsec]
    cjdc, cfidc, cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic: Spark DataFrame Columns
        Columns containing history of fid, magpsf, sigmapsf, magnr, sigmagnr,
        magzpsci, isdiffpos as arrays
    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.
    """
    # Extract last (new) measurement from the concatenated column
    jd = cjdc.apply(lambda x: x[-1])
    fid = cfidc.apply(lambda x: x[-1])
    isdiffpos = cisdiffposc.apply(lambda x: x[-1])

    high_drb = drb.astype(float) > 0.9
    high_classtar = classtar.astype(float) > 0.4
    new_detection = jd.astype(float) - jdstarthist.astype(float) < 5
    small_detection_history = ndethist.astype(float) < 20
    appeared = isdiffpos.astype(str) == 't'
    far_from_mpc = (ssdistnr.astype(float) > 10) | (ssdistnr.astype(float) < 0)

    # galactic plane
    b = SkyCoord(ra.astype(float), dec.astype(float), unit='deg').galactic.b.deg

    awaw_from_galactic_plane = np.abs(b) > 10

    keep_cds = return_list_of_eg_host()

    f_kn = high_drb & high_classtar & new_detection & small_detection_history
    f_kn = f_kn & cdsxmatch.isin(keep_cds) & appeared & far_from_mpc
    f_kn = f_kn & awaw_from_galactic_plane

    # Compute rate and error rate, get magnitude and its error
    rate = np.zeros(len(fid))
    sigma_rate = np.zeros(len(fid))
    mag = np.zeros(len(fid))
    err_mag = np.zeros(len(fid))
    index_mask = np.argwhere(f_kn.values)
    for i, alertID in enumerate(objectId[f_kn]):
        # Spark casts None as NaN
        maskNotNone = ~np.isnan(np.array(cmagpsfc[f_kn].values[i]))
        maskFilter = np.array(cfidc[f_kn].values[i]) == np.array(fid)[f_kn][i]
        m = maskNotNone * maskFilter
        if sum(m) < 2:
            continue
        # DC mag (history + last measurement)
        mag_hist, err_hist = np.array([
            dc_mag(k[0], k[1], k[2], k[3], k[4], k[5], k[6])
            for k in zip(
                cfidc[f_kn].values[i][m],
                cmagpsfc[f_kn].values[i][m],
                csigmapsfc[f_kn].values[i][m],
                cmagnrc[f_kn].values[i][m],
                csigmagnrc[f_kn].values[i][m],
                cmagzpscic[f_kn].values[i][m],
                cisdiffposc[f_kn].values[i][m],
            )
        ]).T

        # remove abnormal values
        mask_outliers = mag_hist < 21
        if sum(mask_outliers) < 2:
            continue
        jd_hist = cjdc[f_kn].values[i][m][mask_outliers]

        if jd_hist[-1] - jd_hist[0] > 0.5:
            # Compute rate
            popt, pcov = curve_fit(
                lambda x, a, b: a * x + b,
                jd_hist,
                mag_hist[mask_outliers],
                sigma=err_hist[mask_outliers],
            )
            rate[index_mask[i]] = popt[0]
            sigma_rate[index_mask[i]] = pcov[0, 0]

        # Grab the last measurement and its error estimate
        mag[index_mask[i]] = mag_hist[-1]
        err_mag[index_mask[i]] = err_hist[-1]

    # filter on rate. rate is 0 where f_kn is already false.
    f_kn = pd.Series(np.array(rate) > 0.3)

    # check the nature of close objects in SDSS catalog
    if f_kn.any():
        no_star = []
        for i in range(sum(f_kn)):
            pos = SkyCoord(
                ra=np.array(ra[f_kn])[i] * u.degree,
                dec=np.array(dec[f_kn])[i] * u.degree
            )
            # for a test on "many" objects, you may wait 1s to stay under the
            # query limit.
            table = SDSS.query_region(pos, fields=['type'],
                                      radius=5 * u.arcsec)
            type_close_objects = []
            if table is not None:
                type_close_objects = table['type']
            # types: 0: UNKNOWN, 1: STAR, 2: GALAXY, 3: QSO, 4: HIZ_QSO,
            # 5: SKY, 6: STAR_LATE, 7: GAL_EM
            to_remove_types = [1, 3, 4, 6]
            no_star.append(
                len(np.intersect1d(type_close_objects, to_remove_types)) == 0
            )
        f_kn.loc[f_kn] = np.array(no_star, dtype=bool)

    return f_kn, rate, sigma_rate, mag, err_mag

def rate_based_kn_candidates_(
        objectId, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jdstarthist, ndethist, cdsxmatch, ra, dec, ssdistnr, cjdc,
        cfidc, cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic,
        cisdiffposc) -> pd.Series:
    """
    Return alerts considered as KN candidates.

    The cuts are based on Andreoni et al. 2021 https://arxiv.org/abs/2104.06352

    If the environment variable KNWEBHOOK is defined and match a webhook url,
    the alerts that pass the filter will be sent to the matching Slack channel.

    Parameters
    ----------
    objectId: Spark DataFrame Column
        Column containing the alert IDs
    rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all: Spark DataFrame Columns
        Columns containing the scores for: 'Early SN Ia',
        'Ia SN vs non-Ia SN', 'SN Ia and Core-Collapse vs non-SN events'
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    jdstarthist: Spark DataFrame Column
        Column containing earliest Julian dates of epoch [days]
    ndethist: Spark DataFrame Column
        Column containing the number of prior detections (theshold of 3 sigma)
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    ra: Spark DataFrame Column
        Column containing the right Ascension of candidate; J2000 [deg]
    dec: Spark DataFrame Column
        Column containing the declination of candidate; J2000 [deg]
    ssdistnr: Spark DataFrame Column
        distance to nearest known solar system object; -999.0 if none [arcsec]
    cjdc, cfidc, cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic: Spark DataFrame Columns
        Columns containing history of fid, magpsf, sigmapsf, magnr, sigmagnr,
        magzpsci, isdiffpos as arrays
    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.
    """
    f_kn, _, _, _, _ = perform_classification(
        objectId, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jdstarthist, ndethist, cdsxmatch, ra, dec, ssdistnr, cjdc,
        cfidc, cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic,
        cisdiffposc
    )

    return f_kn

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def rate_based_kn_candidates(
        objectId, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jdstarthist, ndethist, cdsxmatch, ra, dec, ssdistnr, cjdc,
        cfidc, cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic,
        cisdiffposc) -> pd.Series:
    """
    Return alerts considered as KN candidates.

    The cuts are based on Andreoni et al. 2021 https://arxiv.org/abs/2104.06352

    If the environment variable KNWEBHOOK is defined and match a webhook url,
    the alerts that pass the filter will be sent to the matching Slack channel.

    Parameters
    ----------
    objectId: Spark DataFrame Column
        Column containing the alert IDs
    rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all: Spark DataFrame Columns
        Columns containing the scores for: 'Early SN Ia',
        'Ia SN vs non-Ia SN', 'SN Ia and Core-Collapse vs non-SN events'
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    jdstarthist: Spark DataFrame Column
        Column containing earliest Julian dates of epoch [days]
    ndethist: Spark DataFrame Column
        Column containing the number of prior detections (theshold of 3 sigma)
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    ra: Spark DataFrame Column
        Column containing the right Ascension of candidate; J2000 [deg]
    dec: Spark DataFrame Column
        Column containing the declination of candidate; J2000 [deg]
    ssdistnr: Spark DataFrame Column
        distance to nearest known solar system object; -999.0 if none [arcsec]
    cjdc, cfidc, cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic: Spark DataFrame Columns
        Columns containing history of fid, magpsf, sigmapsf, magnr, sigmagnr,
        magzpsci, isdiffpos as arrays
    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import concat_col
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest')

    >>> to_expand = ['jd', 'fid', 'magpsf', 'sigmapsf', 'magnr', 'sigmagnr', 'magzpsci', 'isdiffpos']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> f = 'fink_filters.filter_rate_based_kn_candidates.filter.rate_based_kn_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    0
    """
    f_kn, rate, sigma_rate, mag, err_mag = perform_classification(
        objectId, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jdstarthist, ndethist, cdsxmatch, ra, dec, ssdistnr, cjdc,
        cfidc, cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic,
        cisdiffposc
    )

    jd = cjdc.apply(lambda x: x[-1])
    fid = cfidc.apply(lambda x: x[-1])

    # galactic plane
    b = SkyCoord(ra.astype(float), dec.astype(float), unit='deg').galactic.b.deg

    # Simplify notations
    if f_kn.any():
        # coordinates
        b = np.array(b)[f_kn]
        ra = Angle(
            np.array(ra.astype(float)[f_kn]) * u.degree
        ).deg
        dec = Angle(
            np.array(dec.astype(float)[f_kn]) * u.degree
        ).deg
        ra_formatted = Angle(ra * u.degree).to_string(
            precision=2, sep=' ', unit=u.hour
        )
        dec_formatted = Angle(dec * u.degree).to_string(
            precision=1, sep=' ', alwayssign=True
        )
        delta_jd_first = np.array(
            jd.astype(float)[f_kn] - jdstarthist.astype(float)[f_kn]
        )

        # scores
        rf_snia_vs_nonia = np.array(rf_snia_vs_nonia.astype(float)[f_kn])
        snn_snia_vs_nonia = np.array(snn_snia_vs_nonia.astype(float)[f_kn])
        snn_sn_vs_all = np.array(snn_sn_vs_all.astype(float)[f_kn])

        # time
        fid = np.array(fid.astype(int)[f_kn])
        jd = np.array(jd)[f_kn]

        # measurements
        mag = mag[f_kn]
        rate = rate[f_kn]
        err_mag = err_mag[f_kn]
        sigma_rate = sigma_rate[f_kn]

    # message for candidates
    for i, alertID in enumerate(objectId[f_kn]):

        # Time since last detection (independently of the band)
        maskNotNone = ~np.isnan(np.array(cmagpsfc[f_kn].values[i]))
        jd_hist_allbands = np.array(np.array(cjdc[f_kn])[i])[maskNotNone]
        delta_jd_last = jd_hist_allbands[-1] - jd_hist_allbands[-2]

        # information to send
        dict_filt = {1: 'g', 2: 'r'}
        alert_text = """
            *New kilonova candidate:* <https://fink-portal.org/{}|{}>
            """.format(alertID, alertID)
        score_text = """
            *Scores:*\n- Early SN Ia: {:.2f}\n- Ia SN vs non-Ia SN: {:.2f}\n- SN Ia and Core-Collapse vs non-SN: {:.2f}
            """.format(rf_snia_vs_nonia[i], snn_snia_vs_nonia[i], snn_sn_vs_all[i])
        time_text = """
            *Time:*\n- {} UTC\n - Time since last detection: {:.1f} days\n - Time since first detection: {:.1f} days
            """.format(Time(jd[i], format='jd').iso, delta_jd_last, delta_jd_first[i])
        measurements_text = """
            *Measurement (band {}):*\n- Apparent magnitude: {:.2f} ± {:.2f} \n- Rate: ({:.2f} ± {:.2f}) mag/day\n
            """.format(dict_filt[fid[i]], mag[i], err_mag[i], rate[i], sigma_rate[i])
        radec_text = """
              *RA/Dec:*\n- [hours, deg]: {} {}\n- [deg, deg]: {:.7f} {:+.7f}
              """.format(ra_formatted[i], dec_formatted[i], ra[i], dec[i])
        galactic_position_text = """
            *Galactic latitude:*\n- [deg]: {:.7f}""".format(b[i])

        tns_text = '*TNS:* <https://www.wis-tns.org/search?ra={}&decl={}&radius=5&coords_unit=arcsec|link>'.format(ra[i], dec[i])
        # message formatting
        blocks = [
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": alert_text
                    },
                ]
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": time_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": score_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": radec_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": measurements_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": galactic_position_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": tns_text
                    },
                ]
            },
        ]

        error_message = """
        {} is not defined as env variable
        if an alert has passed the filter,
        the message has not been sent to Slack
        """
        for url_name in ['KNWEBHOOK', 'KNWEBHOOK_FINK']:
            if (url_name in os.environ):
                requests.post(
                    os.environ[url_name],
                    json={
                        'blocks': blocks,
                        'username': 'Rate-based kilonova bot'
                    },
                    headers={'Content-Type': 'application/json'},
                )
            else:
                log = logging.Logger('Kilonova filter')
                log.warning(error_message.format(url_name))

        ama_in_env = ('KNWEBHOOK_AMA_RATE' in os.environ)

        # Send alerts to amateurs only on Friday
        now = datetime.datetime.utcnow()

        # Monday is 1 and Sunday is 7
        is_friday = (now.isoweekday() == 5)

        if (np.abs(b[i]) > 20) & (mag[i] < 20) & is_friday & ama_in_env:
            requests.post(
                os.environ['KNWEBHOOK_AMA_RATE'],
                json={
                    'blocks': blocks,
                    'username': 'Rate-based kilonova bot'
                },
                headers={'Content-Type': 'application/json'},
            )
        else:
            log = logging.Logger('Kilonova filter')
            log.warning(error_message.format(url_name))

    return f_kn


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
