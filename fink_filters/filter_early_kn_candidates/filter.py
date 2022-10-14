# Copyright 2021-2022 AstroLab Software
# Author: Juliette Vlieghe, Julien Peloton
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
import logging
import os

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from astropy.time import Time
from astroquery.sdss import SDSS

from fink_utils.photometry.conversion import dc_mag
from fink_utils.xmatch.simbad import return_list_of_eg_host

from fink_filters.tester import spark_unit_tests

def perform_classification(drb, classtar, jd, jdstarthist, ndethist, cdsxmatch, fid,
        magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos, ra, dec, roid):
    """
    """
    high_drb = drb.astype(float) > 0.5
    high_classtar = classtar.astype(float) > 0.4
    new_detection = jd.astype(float) - jdstarthist.astype(float) < 0.25
    not_ztf_sso_candidate = roid.astype(int) != 3

    keep_cds = return_list_of_eg_host()

    # Compute DC magnitude
    mag, err_mag = np.array([
        dc_mag(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
        for i in zip(
            np.array(fid),
            np.array(magpsf),
            np.array(sigmapsf),
            np.array(magnr),
            np.array(sigmagnr),
            np.array(magzpsci),
            np.array(isdiffpos))
    ]).T

    f_kn = high_drb & high_classtar & new_detection
    f_kn = f_kn & cdsxmatch.isin(keep_cds) & not_ztf_sso_candidate

    # Containers
    pdf_mangrove = pd.DataFrame()
    host_galaxies = []
    host_alert_separation = []
    abs_mag_candidate = []

    if f_kn.any():
        # load mangrove catalog
        curdir = os.path.dirname(os.path.abspath(__file__))
        mangrove_path = curdir + '/../data/mangrove_filtered.csv'
        pdf_mangrove = pd.read_csv(mangrove_path)

        catalog_mangrove = SkyCoord(
            ra=np.array(pdf_mangrove.ra, dtype=float) * u.degree,
            dec=np.array(pdf_mangrove.dec, dtype=float) * u.degree
        )

        pdf = pd.DataFrame.from_dict(
            {
                'fid': fid[f_kn], 'ra': ra[f_kn],
                'dec': dec[f_kn], 'mag': mag[f_kn],
                'err_mag': err_mag[f_kn]
            }
        )

        # identify galaxy somehow close to each alert. Distances are in Mpc
        idx_mangrove, idxself, _, _ = SkyCoord(
            ra=np.array(pdf.ra, dtype=float) * u.degree,
            dec=np.array(pdf.dec, dtype=float) * u.degree
        ).search_around_sky(catalog_mangrove, 2 * u.degree)

        # cross match
        galaxy_matching = []
        host_galaxies = []
        abs_mag_candidate = []
        host_alert_separation = []
        for i, row in enumerate(pdf.itertuples()):
            # SkyCoord didn't keep the original indexes
            idx_reduced = idx_mangrove[idxself == i]
            abs_mag = np.array(row.mag - 25 - 5 * np.log10(
                pdf_mangrove.loc[idx_reduced, :].lum_dist))

            candidates_number = np.argwhere(
                np.array(
                    (
                        SkyCoord(
                            ra=row.ra * u.degree,
                            dec=row.dec * u.degree
                        ).separation(catalog_mangrove[idx_reduced]).radian < 0.01 / pdf_mangrove.loc[idx_reduced, :].ang_dist
                    ) & (abs_mag > -17) & (abs_mag < -15)
                )
            )
            galaxy_matching.append(len(candidates_number) > 0)

            # save useful information on successful candidates
            if len(candidates_number) > 0:
                host_galaxies.append(idx_reduced[candidates_number[0][0]])
                abs_mag_candidate.append(abs_mag[candidates_number[0][0]])
                host_alert_separation.append(
                    SkyCoord(
                        ra=row.ra * u.degree,
                        dec=row.dec * u.degree
                    ).separation(
                        catalog_mangrove[idx_reduced[candidates_number[0][0]]]
                    ).radian
                )
                # There are sometimes 2 hosts, we currently take the closest
                # to earth.
                # This is the index of catalog dataframe and has nothing to do
                # with galaxies idx.

        f_kn.loc[f_kn] = np.array(galaxy_matching, dtype=bool)

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

    return f_kn, pdf_mangrove, host_galaxies, host_alert_separation, abs_mag_candidate, mag, err_mag

def early_kn_candidates_(
        drb, classtar, jd, jdstarthist, ndethist, cdsxmatch, fid,
        magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos, ra, dec, roid) -> pd.Series:
    """ Return alerts considered as KN candidates from the xmatch with Mangrove

    Note the default `data/mangrove_filtered.csv` catalog is loaded.

    Parameters
    ----------
    drb: Pandas series
        Column containing the Deep-Learning Real Bogus score
    classtar: Pandas series
        Column containing the sextractor score
    jd: Pandas series
        Column containing observation Julian dates at start of exposure [days]
    jdstarthist: Pandas series
        Column containing earliest Julian dates corresponding to ndethist
    ndethist: Pandas series
        Column containing the number of prior detections (theshold of 3 sigma)
    cdsxmatch: Pandas series
        Column containing the cross-match values
    fid: Pandas series
        Column containing filter, 1 for green and 2 for red
    magpsf,sigmapsf: Pandas series
        Columns containing magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr: Pandas series
        Columns containing magnitude of nearest source in reference image
        PSF-catalog within 30 arcsec and 1-sigma error
    magzpsci: Pandas series
        Column containing magnitude zero point for photometry estimates
    isdiffpos: Pandas series
        Column containing:
        t or 1 => candidate is from positive (sci minus ref) subtraction;
        f or 0 => candidate is from negative (ref minus sci) subtraction
    ra: Pandas series
        Column containing the right Ascension of candidate; J2000 [deg]
    dec: Pandas series
        Column containing the declination of candidate; J2000 [deg]
    magpsf: Pandas series
        Column containing the magnitude from PSF-fit photometry [mag]
    roid: Pandas series
        Column containing the Solar System label

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> pdf = pd.read_parquet('datatest')
    >>> classification = early_kn_candidates_(
    ...     pdf['candidate'].apply(lambda x: x['drb']),
    ...     pdf['candidate'].apply(lambda x: x['classtar']),
    ...     pdf['candidate'].apply(lambda x: x['jd']),
    ...     pdf['candidate'].apply(lambda x: x['jdstarthist']),
    ...     pdf['candidate'].apply(lambda x: x['ndethist']),
    ...     pdf['cdsxmatch'],
    ...     pdf['candidate'].apply(lambda x: x['fid']),
    ...     pdf['candidate'].apply(lambda x: x['magpsf']),
    ...     pdf['candidate'].apply(lambda x: x['sigmapsf']),
    ...     pdf['candidate'].apply(lambda x: x['magnr']),
    ...     pdf['candidate'].apply(lambda x: x['sigmagnr']),
    ...     pdf['candidate'].apply(lambda x: x['magzpsci']),
    ...     pdf['candidate'].apply(lambda x: x['isdiffpos']),
    ...     pdf['candidate'].apply(lambda x: x['ra']),
    ...     pdf['candidate'].apply(lambda x: x['dec']),
    ...     pdf['roid'])
    >>> print(pdf[classification]['objectId'].values)
    []
    """
    f_kn, _, _, _, _, _, _ = perform_classification(
        drb, classtar, jd, jdstarthist, ndethist, cdsxmatch, fid,
        magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos, ra, dec, roid
    )

    return f_kn

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def early_kn_candidates(
        objectId, drb, classtar, jd, jdstarthist, ndethist, cdsxmatch, fid,
        magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos, ra, dec, roid,
        field) -> pd.Series:
    """
    Return alerts considered as KN candidates.

    If the environment variable KNWEBHOOK is defined and match a
    webhook url, the alerts that pass the filter will be sent to the matching
    Slack channel.

    Note the default `data/mangrove_filtered.csv` catalog is loaded.

    Parameters
    ----------
    objectId: Spark DataFrame Column
        Column containing the alert IDs
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    jd: Spark DataFrame Column
        Column containing observation Julian dates at start of exposure [days]
    jdstarthist: Spark DataFrame Column
        Column containing earliest Julian dates corresponding to ndethist
    ndethist: Spark DataFrame Column
        Column containing the number of prior detections (theshold of 3 sigma)
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    fid: Spark DataFrame Column
        Column containing filter, 1 for green and 2 for red
    magpsf,sigmapsf: Spark DataFrame Columns
        Columns containing magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr: Spark DataFrame Columns
        Columns containing magnitude of nearest source in reference image
        PSF-catalog within 30 arcsec and 1-sigma error
    magzpsci: Spark DataFrame Column
        Column containing magnitude zero point for photometry estimates
    isdiffpos: Spark DataFrame Column
        Column containing:
        t or 1 => candidate is from positive (sci minus ref) subtraction;
        f or 0 => candidate is from negative (ref minus sci) subtraction
    ra: Spark DataFrame Column
        Column containing the right Ascension of candidate; J2000 [deg]
    dec: Spark DataFrame Column
        Column containing the declination of candidate; J2000 [deg]
    magpsf: Spark DataFrame Column
        Column containing the magnitude from PSF-fit photometry [mag]
    roid: Spark DataFrame Column
        Column containing the Solar System label
    field: Spark DataFrame Column
        Column containing the ZTF field numbers (int)

    Returns
    -------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.

    Examples
    ----------
    >>> from fink_utils.spark.utils import apply_user_defined_filter
    >>> df = spark.read.format('parquet').load('datatest')
    >>> f = 'fink_filters.filter_early_kn_candidates.filter.early_kn_candidates'
    >>> df = apply_user_defined_filter(df, f)
    >>> print(df.count())
    0
    """
    # galactic plane
    gal = SkyCoord(ra.astype(float), dec.astype(float), unit='deg').galactic

    out = perform_classification(
        drb, classtar, jd, jdstarthist, ndethist, cdsxmatch, fid,
        magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos, ra, dec, roid
    )

    f_kn, pdf_mangrove, host_galaxies, host_alert_separation, \
        abs_mag_candidate, mag, err_mag = out

    if f_kn.any():
        # Simplify notations
        b = gal.b.degree[f_kn]
        ra = Angle(
            np.array(ra.astype(float)[f_kn]) * u.degree
        ).deg
        dec = Angle(
            np.array(dec.astype(float)[f_kn]) * u.degree
        ).deg
        ra_formatted = Angle(ra * u.degree).to_string(
            precision=2, sep=' ',
            unit=u.hour
        )
        dec_formatted = Angle(dec * u.degree).to_string(
            precision=1, sep=' ',
            alwayssign=True
        )
        delta_jd_first = np.array(
            jd.astype(float)[f_kn] - jdstarthist.astype(float)[f_kn]
        )
        # Redefine notations relative to candidates
        fid = np.array(fid)[f_kn]
        jd = np.array(jd)[f_kn]
        mag = mag[f_kn]
        err_mag = err_mag[f_kn]
        field = field[f_kn]

    dict_filt = {1: 'g', 2: 'r'}
    for i, alertID in enumerate(objectId[f_kn]):
        # information to send
        alert_text = """
            *New kilonova candidate:* <https://fink-portal.org/{}|{}>
            """.format(alertID, alertID)
        time_text = """
            *Time:*\n- {} UTC\n - Time since first detection: {:.1f} hours
            """.format(Time(jd[i], format='jd').iso, delta_jd_first[i] * 24)
        measurements_text = """
            *Measurement (band {}):*\n- Apparent magnitude: {:.2f} ± {:.2f}
            """.format(dict_filt[fid[i]], mag[i], err_mag[i])
        host_text = """
            *Presumed host galaxy:*\n- HyperLEDA Name: {:s}\n- 2MASS XSC Name: {:s}\n- Luminosity distance: ({:.2f} ± {:.2f}) Mpc\n- RA/Dec: {:.7f} {:+.7f}\n- log10(Stellar mass/Ms): {:.2f}
            """.format(
            pdf_mangrove.loc[host_galaxies[i], 'HyperLEDA_name'][2:-1],
            pdf_mangrove.loc[host_galaxies[i], '2MASS_name'][2:-1],
            pdf_mangrove.loc[host_galaxies[i], 'lum_dist'],
            pdf_mangrove.loc[host_galaxies[i], 'dist_err'],
            pdf_mangrove.loc[host_galaxies[i], 'ra'],
            pdf_mangrove.loc[host_galaxies[i], 'dec'],
            pdf_mangrove.loc[host_galaxies[i], 'stellarmass'],
        )
        crossmatch_text = """
        *Cross-match: *\n- Alert-host distance: {:.2f} kpc\n- Absolute magnitude: {:.2f}
        """.format(
            host_alert_separation[i] * pdf_mangrove.loc[
                host_galaxies[i], 'ang_dist'] * 1000,
            abs_mag_candidate[i],
        )
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
                        "text": host_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": radec_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": crossmatch_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": galactic_position_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": measurements_text
                    },
                    {
                        "type": "mrkdwn",
                        "text": tns_text
                    },
                ]
            },
        ]

        # Standard channels
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
                        'username': 'Cross-match-based kilonova bot'
                    },
                    headers={'Content-Type': 'application/json'},
                )
            else:
                log = logging.Logger('Kilonova filter')
                log.warning(error_message.format(url_name))

        # Grandma amateur channel
        ama_in_env = ('KNWEBHOOK_AMA_GALAXIES' in os.environ)

        # Send alerts to amateurs only on Friday
        now = datetime.datetime.utcnow()

        # Monday is 1 and Sunday is 7
        is_friday = (now.isoweekday() == 5)

        if (np.abs(b[i]) > 20) & (mag[i] < 20) & is_friday & ama_in_env:
            requests.post(
                os.environ['KNWEBHOOK_AMA_GALAXIES'],
                json={
                    'blocks': blocks,
                    'username': 'Cross-match-based kilonova bot'
                },
                headers={'Content-Type': 'application/json'},
            )
        else:
            log = logging.Logger('Kilonova filter')
            log.warning(error_message.format('KNWEBHOOK_AMA_GALAXIES'))

        # DWF channel and requirements
        dwf_ztf_fields = [1525, 530, 482, 1476, 388, 1433]
        dwf_in_env = ('KNWEBHOOK_DWF' in os.environ)
        if (int(field.values[i]) in dwf_ztf_fields) and dwf_in_env:
            requests.post(
                os.environ['KNWEBHOOK_DWF'],
                json={
                    'blocks': blocks,
                    'username': 'kilonova bot'
                },
                headers={'Content-Type': 'application/json'},
            )
        else:
            log = logging.Logger('Kilonova filter')
            log.warning(error_message.format('KNWEBHOOK_DWF'))

    return f_kn


if __name__ == "__main__":
    """ Execute the test suite """

    # Run the test suite
    globs = globals()
    spark_unit_tests(globs)
