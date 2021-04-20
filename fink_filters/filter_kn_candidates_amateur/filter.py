# Copyright 2019-2020 AstroLab Software
# Author: Juliette Vlieghe
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
import requests
import logging
import os

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from astropy.time import Time

from fink_science.conversion import dc_mag


@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def early_kn_candidates(
        objectId, drb, classtar, jd, jdstarthist, ndethist, cdsxmatch, fid,
        magpsf, sigmapsf, magnr, sigmagnr, magzpsci, isdiffpos, ra, dec, roid,
        mangrove_path=None) -> pd.Series:
    """ Return alerts considered as KN candidates.
    If the environment variable KNWEBHOOK_MANGROVE is defined and match a
    webhook url, the alerts that pass the filter will be sent to the matching
    Slack channel.

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
    mangrove_path: Spark DataFrame Column, optional
        Path to the Mangrove file. Default is None, in which case
        `data/mangrove_filtered.csv` is loaded.

    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.
    """

    high_drb = drb.astype(float) > 0.5
    high_classtar = classtar.astype(float) > 0.4
    new_detection = jd.astype(float) - jdstarthist.astype(float) < 0.25
    not_sso_candidate = (roid.astype(int) != 2) & (roid.astype(int) != 3)

    # galactic plane
    gal = SkyCoord(ra.astype(float), dec.astype(float), unit='deg').galactic
    outside_galactic_plane = np.abs(gal.b.degree) > 20

    list_simbad_galaxies = [
        "galaxy",
        "Galaxy",
        "EmG",
        "Seyfert",
        "Seyfert_1",
        "Seyfert_2",
        "BlueCompG",
        "StarburstG",
        "LSB_G",
        "HII_G",
        "High_z_G",
        "GinPair",
        "GinGroup",
        "BClG",
        "GinCl",
        "PartofG",
    ]

    keep_cds = \
        ["Unknown", "Transient", "Fail"] + list_simbad_galaxies

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
    low_app_magnitude = mag.astype(float) < 20

    f_kn = high_drb & high_classtar & new_detection
    f_kn = f_kn & cdsxmatch.isin(keep_cds) & not_sso_candidate
    f_kn = f_kn & low_app_magnitude & outside_galactic_plane

    if f_kn.any():
        # load mangrove catalog
        if mangrove_path is not None:
            pdf_mangrove = pd.read_csv(mangrove_path.values[0])
        else:
            curdir = os.path.dirname(os.path.abspath(__file__))
            mangrove_path = curdir + '/data/mangrove_filtered.csv'
            pdf_mangrove = pd.read_csv(mangrove_path)
        catalog_mangrove = SkyCoord(
            ra=np.array(pdf_mangrove.ra, dtype=np.float) * u.degree,
            dec=np.array(pdf_mangrove.dec, dtype=np.float) * u.degree
        )

        pdf = pd.DataFrame.from_dict({'fid': fid[f_kn], 'ra': ra[f_kn],
                                      'dec': dec[f_kn], 'mag': mag[f_kn],
                                      'err_mag': err_mag[f_kn]})

        # identify galaxy somehow close to each alert. Distances are in Mpc
        idx_mangrove, idxself, _, _ = SkyCoord(
            ra=np.array(pdf.ra, dtype=np.float) * u.degree,
            dec=np.array(pdf.dec, dtype=np.float) * u.degree
            ).search_around_sky(catalog_mangrove, 2*u.degree)

        # cross match
        galaxy_matching = []
        for i, row in enumerate(pdf.itertuples()):
            # SkyCoord didn't keep the original indexes
            idx_reduced = idx_mangrove[idxself == i]
            abs_mag = row.mag-1-5*np.log10(
                pdf_mangrove.loc[idx_reduced, :].lum_dist)

            # cross-match on position. We take a radius of 50 kpc
            galaxy_matching.append((
                (SkyCoord(
                    ra=row.ra*u.degree,
                    dec=row.dec*u.degree
                ).separation(catalog_mangrove[idx_reduced]).radian
                    < 0.05/pdf_mangrove.loc[idx_reduced, :].ang_dist)

                & (abs_mag > 15) & (abs_mag < 17)
            ).any())

        f_kn[f_kn] = galaxy_matching

    if 'KNWEBHOOK_AMA' in os.environ:
        if f_kn.any():
            # Simplify notations
            b = gal.b.degree[f_kn]
            ra = Angle(
                np.array(ra.astype(float)[f_kn]) * u.degree
            ).to_string(precision=1)
            dec = Angle(
                np.array(dec.astype(float)[f_kn]) * u.degree
            ).to_string(precision=1)
            delta_jd_first = np.array(
                jd.astype(float)[f_kn] - jdstarthist.astype(float)[f_kn]
            )
            # Redefine jd & fid relative to candidates
            fid = np.array(fid)[f_kn]
            jd = np.array(jd)[f_kn]
            mag = mag[f_kn]
            err_mag = err_mag[f_kn]

        dict_filt = {1: 'g', 2: 'r'}
        for i, alertID in enumerate(objectId[f_kn]):
            # information to send
            alert_text = """
                *New kilonova candidate:* <http://134.158.75.151:24000/{}|{}>
                """.format(alertID, alertID)
            time_text = """
                *Time:*\n- {} UTC\n - Time since first detection: {:.1f} days
                """.format(Time(jd[i], format='jd').iso, delta_jd_first[i])
            measurements_text = """
                *Measurement (band {}):*\n- Apparent magnitude: {:.2f} ± {:.2f}\n- Absolute magnitude: {}
                """.format(dict_filt[fid[i]], mag[i], err_mag[i], ' ',)
            host_text = """
                *Presumed host galaxy (closest candidate):*\n- Name: {}\n- Luminosity distance: {}\n- Galactic latitude:\t{}\n
                """.format('', ' ', ' ')
            position_text = """
            *Position:*\n- Right ascension:\t {}\n- Declination:\t\t\t{}\n- Galactic latitude:\t{}
            """.format(ra[i], dec[i], b[i])
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
                            "text": measurements_text
                        },
                        {
                            "type": "mrkdwn",
                            "text": position_text
                        },
                        {
                            "type": "mrkdwn",
                            "text": host_text
                        },
                    ]
                },
            ]

            requests.post(
                os.environ['KNWEBHOOK_AMA'],
                json={
                    'blocks': blocks,
                    'username': 'Cross-match-based kilonova bot'
                },
                headers={'Content-Type': 'application/json'},
            )
    else:
        log = logging.Logger('Kilonova filter')
        msg = """
        KNWEBHOOK_AMA is not defined as env variable
        if an alert has passed the filter,
        the message has not been sent to Slack
        """
        log.warning(msg)

    return f_kn
