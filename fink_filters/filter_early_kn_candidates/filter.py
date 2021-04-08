from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import numpy as np
import pandas as pd
import requests, logging
import os
import h5py

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def early_kn_candidates(objectId, drb, classtar, jd, jdstarthist, ndethist, 
                  cdsxmatch, ra, dec, magpsf, mangrove_path=None) -> pd.Series:
    """ Return alerts considered as KN candidates.
    If the environment variable KNWEBHOOK is defined and match a webhook url,
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

    f_kn = high_drb & high_classtar & new_detection & small_detection_history
    f_kn = f_kn & cdsxmatch.isin(keep_cds)
    
    # cross match with Mangrove catalog
    if f_kn.any():
        if mangrove_path is not None:
            pdf_mangrove = pd.read_csv(mangrove_path.values[0])
        else:
            curdir = os.path.dirname(os.path.abspath(__file__))
            mangrove_path = curdir + '/data/mangrove_filtered.csv'
            pdf_mangrove = pd.read_csv(mangrove_path)
        pdf = pd.DataFrame.from_dict({'ra':ra,'dec':dec,'magpsf':magpsf})
        galaxy_matching = pdf[f_kn].apply(
            lambda row:
                ((np.sqrt(np.square(row.dec-pdf_mangrove.dec)+np.square(row.ra-pdf_mangrove.ra)
                    )<0.1/pdf_mangrove.dist)
                 & (row.magpsf+5-5*np.log10(pdf_mangrove.dist)>16-1)
                 & (row.magpsf+5-5*np.log10(pdf_mangrove.dist)<16+1)
                ).any(),
            axis=1
        )
        f_kn[f_kn] = galaxy_matching
        
    
    if 'KNWEBHOOK' in os.environ:
        for alertID in objectId[f_kn]:
            slacktext = f'new kilonova candidate alert: \n<http://134.158.75.151:24000/{alertID}>'
            requests.post(
                os.environ['KNWEBHOOK'],
                json={'text':slacktext, 'username':'mangrove_kilonova_bot'},
                headers={'Content-Type': 'application/json'},
            )
    else:
        log = logging.Logger('Kilonova filter')
        log.warning('KNWEBHOOK is not defined as env variable -- if an alert passed the filter, message has not been sent to Slack')
    
    return f_kn




def get_mangrove_pdf(path):
    """
    create a pandas dataframe needed in early_kn_candidates from the hdf5 
    Mangrove catalog.
    
    early_kn_candidate loads the pre-filtered csv file, the aim of this 
    function is to create a new csv file if the catalog is updated or the range change.

    Parameters
    ----------
    path : string
        path to Mangrove hdf5 catalog.

    Returns
    -------
    pdf_mangrove : pandas DataFrame
        Dataframe containing the galaxy indexes, their right ascension, 
        declination and distance to earth.

    """
    pdf_mangrove = pd.DataFrame(np.array(
        h5py.File(path,'r')['__astropy_table__']
    ))
    pdf_mangrove = pdf_mangrove.loc[:,['idx','RA','dec','dist']]
    pdf_mangrove = pdf_mangrove[pdf_mangrove.dist<230]
    pdf_mangrove.rename(columns={'idx':'galaxy_idx','RA':'ra'}, inplace=True)
    pdf_mangrove.reset_index(inplace=True, drop=True)
    return pdf_mangrove