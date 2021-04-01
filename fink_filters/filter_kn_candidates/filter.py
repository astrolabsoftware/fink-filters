from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import BooleanType

import pandas as pd
import requests
import os
import logging

@pandas_udf(BooleanType(), PandasUDFType.SCALAR)
def kn_candidates(objectId, knscore, drb, classtar, jd, jdstarthist, ndethist, cdsxmatch) -> pd.Series:
    """ Return alerts considered as KN candidates.
    If the environment variable KNWEBHOOK is defined and match a webhook url,
    the alerts that pass the filter will be sent to the matching Slack channel.
    
    Parameters
    ----------
    objectId: Spark DataFrame Column
        Column containing the alert IDs
    cdsxmatch: Spark DataFrame Column
        Column containing the cross-match values
    drb: Spark DataFrame Column
        Column containing the Deep-Learning Real Bogus score
    classtar: Spark DataFrame Column
        Column containing the sextractor score
    knscore: Spark DataFrame Column
        Column containing the kilonovae score
    jd: Spark DataFrame Column
        Column containing observation Julian dates at start of exposure [days]
    jdstarthist: Spark DataFrame Column
        Column containing earliest Julian dates of epoch corresponding to ndethist [days]
    ndethist: Spark DataFrame Column
        Column containing the number of prior detections (with a theshold of 3 sigma)
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
        for alertID in objectId[f_kn]:
            slacktext = f'new kilonova candidate alert: \n<http://134.158.75.151:24000/{alertID}>'
            requests.post(
                os.environ['KNWEBHOOK'],
                json={'text':slacktext, 'username':'kilonova_bot'},
                headers={'Content-Type': 'application/json'},
            )
    else:
        log = logging.Logger('Kilonova filter')
        log.warning('KNWEBHOOK is not defined as env variable -- if an alert has passed the filter, the message has not been sent to Slack')

    return f_kn
