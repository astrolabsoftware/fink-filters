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

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from astropy.time import Time

from fink_utils.photometry.conversion import dc_mag
from fink_utils.xmatch.simbad import return_list_of_eg_host

from fink_filters.tester import spark_unit_tests

def kn_candidates_(
        rf_kn_vs_nonkn, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jd, jdstarthist, ndethist, cdsxmatch) -> pd.Series:
    """ Return alerts considered as KN candidates.

    Parameters
    ----------
    rf_kn_vs_nonkn, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all: Pandas series
        Columns containing the scores for: 'Kilonova', 'Early SN Ia',
        'Ia SN vs non-Ia SN', 'SN Ia and Core-Collapse vs non-SN events'
    drb: Pandas series
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    jd: Pandas series
    jdstarthist: Pandas series
        Column containing earliest Julian dates of epoch [days]
    ndethist: Pandas series
        Column containing the number of prior detections (theshold of 3 sigma)
    cdsxmatch: Pandas series
        Column containing the cross-match values

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = kn_candidates_(
    ...     pdf['rf_kn_vs_nonkn'],
    ...     pdf['rf_snia_vs_nonia'],
    ...     pdf['snn_snia_vs_nonia'],
    ...     pdf['snn_sn_vs_all'],
    ...     pdf['candidate'].apply(lambda x: x['drb']),
    ...     pdf['candidate'].apply(lambda x: x['classtar']),
    ...     pdf['candidate'].apply(lambda x: x['jd']),
    ...     pdf['candidate'].apply(lambda x: x['jdstarthist']),
    ...     pdf['candidate'].apply(lambda x: x['ndethist']),
    ...     pdf['cdsxmatch'])
    >>> print(pdf[classification]['objectId'].values)
    []
    """
    high_knscore = rf_kn_vs_nonkn.astype(float) > 0.5
    high_drb = drb.astype(float) > 0.5
    high_classtar = classtar.astype(float) > 0.4
    new_detection = jd.astype(float) - jdstarthist.astype(float) < 5
    small_detection_history = ndethist.astype(float) < 20

    keep_cds = return_list_of_eg_host()

    f_kn = high_knscore & high_drb & high_classtar & new_detection
    f_kn = f_kn & small_detection_history & cdsxmatch.isin(keep_cds)

    return f_kn


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def kn_candidates(
        objectId, rf_kn_vs_nonkn, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jdstarthist, ndethist, cdsxmatch, ra, dec, cjdc, cfidc,
        cmagpsfc, csigmapsfc, cmagnrc, csigmagnrc, cmagzpscic, cisdiffposc) -> pd.Series:
    """ Pandas UDF of kn_candidates_ for Spark

    If the environment variable KNWEBHOOK is defined and match a webhook url,
    the alerts that pass the filter will be sent to the matching Slack channel.

    Parameters
    ----------
    objectId: Spark DataFrame Column
        Column containing the alert IDs
    rf_kn_vs_nonkn, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all: Spark DataFrame Columns
        Columns containing the scores for: 'Kilonova', 'Early SN Ia',
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

    >>> to_expand = [
    ...    'jd', 'fid', 'magpsf', 'sigmapsf',
    ...    'magnr', 'sigmagnr', 'magzpsci', 'isdiffpos']

    >>> prefix = 'c'
    >>> for colname in to_expand:
    ...    df = concat_col(df, colname, prefix=prefix)

    # quick fix for https://github.com/astrolabsoftware/fink-broker/issues/457
    >>> for colname in to_expand:
    ...    df = df.withColumnRenamed('c' + colname, 'c' + colname + 'c')

    >>> f = 'fink_filters.filter_kn_candidates.filter.kn_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    0
    """
    # Extract last (new) measurement from the concatenated column
    jd = cjdc.apply(lambda x: x[-1])
    fid = cfidc.apply(lambda x: x[-1])

    f_kn = kn_candidates_(
        rf_kn_vs_nonkn, rf_snia_vs_nonia, snn_snia_vs_nonia, snn_sn_vs_all, drb,
        classtar, jd, jdstarthist, ndethist, cdsxmatch
    )

    if f_kn.any():
        # Galactic latitude transformation
        b = SkyCoord(
            np.array(ra[f_kn], dtype=float),
            np.array(dec[f_kn], dtype=float),
            unit='deg'
        ).galactic.b.deg

        # Simplify notations
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
        rf_kn_vs_nonkn = np.array(rf_kn_vs_nonkn.astype(float)[f_kn])
        rf_snia_vs_nonia = np.array(rf_snia_vs_nonia.astype(float)[f_kn])
        snn_snia_vs_nonia = np.array(snn_snia_vs_nonia.astype(float)[f_kn])
        snn_sn_vs_all = np.array(snn_sn_vs_all.astype(float)[f_kn])

        # Redefine jd & fid relative to candidates
        fid = np.array(fid.astype(int)[f_kn])
        jd = np.array(jd)[f_kn]

    dict_filt = {1: 'g', 2: 'r'}
    for i, alertID in enumerate(objectId[f_kn]):
        # Careful - Spark casts None as NaN!
        maskNotNone = ~np.isnan(np.array(cmagpsfc[f_kn].values[i]))

        # Time since last detection (independently of the band)
        jd_hist_allbands = np.array(np.array(cjdc[f_kn])[i])[maskNotNone]
        delta_jd_last = jd_hist_allbands[-1] - jd_hist_allbands[-2]

        filt = fid[i]
        maskFilter = np.array(cfidc[f_kn].values[i]) == filt
        m = maskNotNone * maskFilter
        if sum(m) < 2:
            continue
        # DC mag (history + last measurement)
        mag_hist, err_hist = np.array([
            dc_mag(k[0], k[1], k[2], k[3], k[4], k[5], k[6])
            for k in zip(
                cfidc[f_kn].values[i][m][-2:],
                cmagpsfc[f_kn].values[i][m][-2:],
                csigmapsfc[f_kn].values[i][m][-2:],
                cmagnrc[f_kn].values[i][m][-2:],
                csigmagnrc[f_kn].values[i][m][-2:],
                cmagzpscic[f_kn].values[i][m][-2:],
                cisdiffposc[f_kn].values[i][m][-2:],
            )
        ]).T

        # Grab the last measurement and its error estimate
        mag = mag_hist[-1]
        err_mag = err_hist[-1]

        # Compute rate only if more than 1 measurement available
        if len(mag_hist) > 1:
            jd_hist = cjdc[f_kn].values[i][m]

            # rate is between `last` and `last-1` measurements only
            dmag = mag_hist[-1] - mag_hist[-2]
            dt = jd_hist[-1] - jd_hist[-2]
            rate = dmag / dt
            error_rate = np.sqrt(err_hist[-1]**2 + err_hist[-2]**2) / dt

        # information to send
        alert_text = """
            *New kilonova candidate:* <https://fink-portal.org/{}|{}>
            """.format(alertID, alertID)
        knscore_text = "*Kilonova score:* {:.2f}".format(rf_kn_vs_nonkn[i])
        score_text = """
            *Other scores:*\n- Early SN Ia: {:.2f}\n- Ia SN vs non-Ia SN: {:.2f}\n- SN Ia and Core-Collapse vs non-SN: {:.2f}
            """.format(rf_snia_vs_nonia[i], snn_snia_vs_nonia[i], snn_sn_vs_all[i])
        time_text = """
            *Time:*\n- {} UTC\n - Time since last detection: {:.1f} days\n - Time since first detection: {:.1f} days
            """.format(Time(jd[i], format='jd').iso, delta_jd_last, delta_jd_first[i])
        measurements_text = """
            *Measurement (band {}):*\n- Apparent magnitude: {:.2f} ± {:.2f} \n- Rate: ({:.2f} ± {:.2f}) mag/day\n
            """.format(dict_filt[fid[i]], mag, err_mag, rate, error_rate)
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
                    {
                        "type": "mrkdwn",
                        "text": knscore_text
                    }
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
                        'username': 'Classifier-based kilonova bot'
                    },
                    headers={'Content-Type': 'application/json'},
                )
            else:
                log = logging.Logger('Kilonova filter')
                log.warning(error_message.format(url_name))

        ama_in_env = ('KNWEBHOOK_AMA_CL' in os.environ)

        # Send alerts to amateurs only on Friday
        now = datetime.datetime.utcnow()

        # Monday is 1 and Sunday is 7
        is_friday = (now.isoweekday() == 5)

        if (np.abs(b[i]) > 20) & (mag < 20) & is_friday & ama_in_env:
            requests.post(
                os.environ['KNWEBHOOK_AMA_CL'],
                json={
                    'blocks': blocks,
                    'username': 'Classifier-based kilonova bot'
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
