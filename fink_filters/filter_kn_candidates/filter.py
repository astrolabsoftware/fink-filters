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
import os
import logging

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u

from fink_science.conversion import dc_mag

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def kn_candidates(objectId, knscore, drb, classtar, jd, jdstarthist, ndethist, 
                  cdsxmatch, fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, 
                  isdiffpos, ra, dec, cjd, cfid, cmagpsf, csigmapsf, cmagnr, 
                  csigmagnr, cmagzpsci, cisdiffpos,) -> pd.Series:
    """ Return alerts considered as KN candidates.
    If the environment variable KNWEBHOOK is defined and match a webhook url,
    the alerts that pass the filter will be sent to the matching Slack channel.
    
    Parameters
    ----------
    objectId: Spark DataFrame Column
        Column containing the alert IDs
    knscore: Spark DataFrame Column
        Column containing the kilonovae scores given by the classifier
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
    cjd, cfid, cmagpsf, csigmapsf, cmagnr, csigmagnr, cmagzpsci: Spark DataFrame Columns
        Columns containing history of fid, magpsf, sigmapsf, magnr, sigmagnr, magzpsci, 
        isdiffpos as arrays
    Returns
    ----------
    out: pandas.Series of bool
        Return a Pandas DataFrame with the appropriate flag:
        false for bad alert, and true for good alert.
    """
    
    high_knscore = knscore.astype(float) > 0.5
    high_drb = drb.astype(float) > 0.5
    high_classtar = classtar.astype(float) > 0.4
    new_detection = jd.astype(float) - jdstarthist.astype(float) < 20
    small_detection_history = ndethist.astype(float) < 20
    
    
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

    f_kn = high_knscore & high_drb & high_classtar & new_detection
    f_kn = f_kn & small_detection_history & cdsxmatch.isin(keep_cds)
    
    if 'KNWEBHOOK' in os.environ:
        if f_kn.any():
            # galactic latitude
            b = SkyCoord(
                np.array(ra[f_kn], dtype=float), 
                np.array(dec[f_kn],dtype=float), 
                unit='deg').galactic.b.to_string(unit=u.degree,precision=1)
            
            # apparent magnitude
            mag, _ = np.array([
                dc_mag(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
                for i in zip(
                    np.array(fid[f_kn]),
                    np.array(magpsf[f_kn]),
                    np.array(sigmapsf[f_kn]),
                    np.array(magnr[f_kn]),
                    np.array(sigmagnr[f_kn]),
                    np.array(magzpsci[f_kn]),
                    np.array(isdiffpos[f_kn]))
                ]).T
            
            # simplify notations
            ra = Angle(np.array(ra.astype(float)[f_kn])*u.degree).to_string(precision=1)
            dec = Angle(np.array(dec.astype(float)[f_kn])*u.degree).to_string(precision=1)
            delta_jd = np.array(jd.astype(float)[f_kn]-jdstarthist.astype(float)[f_kn])
            knscore = np.array(knscore.astype(float)[f_kn])
            fid = np.array(fid.astype(int)[f_kn])
        
        dict_filt={1:'g',2:'r'}
        for i, alertID in enumerate(objectId[f_kn]):
            # Get rates
            maskNotNone = np.array(np.array(cmagpsf[f_kn])[i]) != None
            for filt in [1, 2]:
                maskFilter = np.array(np.array(cfid[f_kn])[i]) == filt
                m = maskNotNone * maskFilter
                # DC mag history
                mag_hist, _ = np.array([
                    dc_mag(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
                    for i in zip(
                        np.array(np.array(cfid[f_kn])[i])[m],
                        np.array(np.array(cmagpsf[f_kn])[i])[m],
                        np.array(np.array(csigmapsf[f_kn])[i])[m],
                        np.array(np.array(cmagnr[f_kn])[i])[m],
                        np.array(np.array(csigmagnr[f_kn])[i])[m],
                        np.array(np.array(cmagzpsci[f_kn])[i])[m],
                        np.array(np.array(cisdiffpos[f_kn])[i])[m])
                    ]).T
                rate = {1:float('nan'),2:float('nan')}
                jd_hist = np.array(np.array(cjd[f_kn])[i])[m]
                if filt == fid[i]: 
                    if len(m)>0:
                        rate[filt] = (mag[i]-mag_hist[-1])/(jd[i]-jd_hist[-1])
                elif len(m)>1:
                        rate[filt] = (mag_hist[-1]-mag_hist[-2])/(jd_hist[-1]-jd_hist[-2])
            
            # message
            position_text="*Position:*\
                \n- Right ascension:\t {}\n- Declination:\t\t\t{}\n- Galactic latitude:\t{}\n"\
                .format(ra[i], dec[i], b[i])
            
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*New kilonova candidate:* <http://134.158.75.151:24000/{alertID}|{alertID}>"
                    }
                 },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": "*Kilonova score:* {:.2f}".format(knscore[i])
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Apparent magnitude (band {}):* {:.2f}\n"\
                            .format(dict_filt[fid[i]], mag[i])
                        },
                        {
                            "type": "mrkdwn",
                            "text": position_text
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Rate:*\n- Band g: {:.2f} mag/day\n- Band r: {:.2f} mag/day"\
                            .format(rate[1],rate[2])
                        },
                        {
                            "type": "mrkdwn",
                            "text": "*Time since first detection:* {:.1f} days"\
                            .format(delta_jd[i])
                        },
                    ]
                },
            ]
            requests.post(
                os.environ['KNWEBHOOK'],
                json={'blocks':blocks, 'username':'Classifier-based kilonova bot'},
                headers={'Content-Type': 'application/json'},
            )
    else:
        log = logging.Logger('Kilonova filter')
        log.warning('KNWEBHOOK is not defined as env variable -- if an alert \
                    has passed the filter, the message has not been sent to Slack')

    return f_kn
