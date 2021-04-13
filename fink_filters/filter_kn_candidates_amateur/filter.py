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
import requests, logging
import os

from astropy.coordinates import SkyCoord
from astropy import units as u

from fink_science.conversion import dc_mag

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def early_kn_candidates(objectId, drb, classtar, jd, jdstarthist, ndethist, 
                cdsxmatch, fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, 
                isdiffpos, ra, dec, mangrove_path=None) -> pd.Series:
    """ Return alerts considered as KN candidates and suitable for amateur observation.
    This filter is similar to early_kn_candidate, with additional cuts.
    If the environment variable KNWEBHOOK_AMA is defined and match a webhook url,
    the alerts that pass the filter will be sent to the matching Slack channel.
    
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
        Column containing earliest Julian dates of epoch corresponding to ndethist [days]
    ndethist: Spark DataFrame Column
        Column containing the number of prior detections (with a theshold of 3 sigma)
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    fid: Spark DataFrame Column
        Column containing filter, 1 for green and 2 for red
    magpsf,sigmapsf: Spark DataFrame Columns
        Columns containing magnitude from PSF-fit photometry, and 1-sigma error
    magnr,sigmagnr: Spark DataFrame Columns
        Columns containing magnitude of nearest source in reference image PSF-catalog
        within 30 arcsec and 1-sigma error
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
    new_detection = jd.astype(float) - jdstarthist.astype(float) < 20
    small_detection_history = ndethist.astype(float) < 20
    
    # galactic plane
    gal = SkyCoord(ra.astype(float), dec.astype(float), unit='deg').galactic
    outside_galactic_plane = np.abs(gal.b.degree)>20
    
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
        ["Unknown", "Transient","Fail"] + list_simbad_galaxies
    
    # apparent magnitude
    mag, _ = np.array([
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
    
    # apply preliminary cuts to reduce the computations needed for cross-match
    f_kn = high_drb & high_classtar & new_detection & small_detection_history
    f_kn = f_kn & cdsxmatch.isin(keep_cds) & low_app_magnitude & outside_galactic_plane
    
    # cross match with Mangrove catalog. Distances are in Mpc
    if f_kn.any():
        # mangrove catalog
        if mangrove_path is not None:
            pdf_mangrove = pd.read_csv(mangrove_path.values[0])
        else:
            curdir = os.path.dirname(os.path.abspath(__file__))
            mangrove_path = curdir + '/data/mangrove_filtered.csv'
            pdf_mangrove = pd.read_csv(mangrove_path)
        catalog_mangrove = SkyCoord(
            ra =np.array(pdf_mangrove.ra, dtype=np.float) * u.degree,
            dec=np.array(pdf_mangrove.dec, dtype=np.float) * u.degree
        )
        # cross-match   
        pdf = pd.DataFrame.from_dict({'fid':fid,'ra':ra,'dec':dec,'mag':mag})
        galaxy_matching = pdf[f_kn].apply(
            lambda row:
                (
                    # cross-match on position.
                    (SkyCoord(
                        ra = row.ra*u.degree, 
                        dec = row.dec*u.degree
                    ).separation(catalog_mangrove).radian<0.1/pdf_mangrove.ang_dist)
                    # if filter is r the cuts on the absolute magnitude do not apply.
                    # this would leave too much alerts, but we most alerts come in pair 
                    # (one band g, one band r), so we can consider that we will get the alert if 
                    # the condition is verified in g band.
                    & (
                        #(row.fid==2) |
                          (row.mag-1-5*np.log10(pdf_mangrove.lum_dist)>16-0.5)
                        & (row.mag-1-5*np.log10(pdf_mangrove.lum_dist)<16+0.5)
                    )
                ).any(),
        axis=1
        )
        f_kn[f_kn] = galaxy_matching

        
    # send alerts to slack
    if 'KNWEBHOOK_AMA' in os.environ:
        for alertID in objectId[f_kn]:
            slacktext = f'new kilonova candidate alert: \n<http://134.158.75.151:24000/{alertID}>'
            requests.post(
                os.environ['KNWEBHOOK_AMA'],
                json={'text':slacktext, 'username':'kilonova_candidates_bot'},
                headers={'Content-Type': 'application/json'},
            )
    else:
        log = logging.Logger('Kilonova filter')
        log.warning('KNWEBHOOK_AMA is not defined as env variable\
        - if an alert passed the filter, message has not been sent to Slack')
    
    return f_kn