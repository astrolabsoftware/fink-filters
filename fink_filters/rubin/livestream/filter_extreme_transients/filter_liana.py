import pandas as pd
import fink_filters.rubin.blocks as fb
import fink_filters.rubin.utils as fu

'''
Criteria for Extreme transient project
- Abs mag at peak -16 to +6
- Rising rate 0.1-1 mag/day
- Decline rate after peak 0.3-1 mag/day
- g-r at peak -0.5 to +0.8
- duration above half peak light <10 days

Something like this(??)
f_extreme_transients = (f_extragalactic & f_bright & f_faint or f_is_rising & f_sampling or f_colour or f_is_fading or f_duration)
'''

# fink-filters/fink_filters/rubin/utils.py
# fink-filters/fink_filters/rubin/blocks.py

def get_valid_rate(mag, filt): # from fink_filters/ztf/filter_orphan_grb_candidates
    """Try to constrain the rate between the 2nd and 3rd measurements

    case 1: the measurements are taken with the same filter
        - mag[2] - mag[1] > 0.0 (becomes fainter)
    case 2: filt(1)=g, filt(2)=r
        - mag[1] - mag[2] <= 0.3 (the difference is smaller than the baseline g-r = 0.3)
    case 1: filt(1)=r, filt(2)=g
        - mag[2] - mag[1] > 0.0 (no real constraints...)

    """
    v = lambda val, mag: val[~np.isnan(mag)]
    filt2nd = v(filt, mag)[1]
    filt3rd = v(filt, mag)[2]

    if filt2nd == filt3rd:
        cond = (v(mag, mag)[2] - v(mag, mag)[1]) > 0.0
    elif filt3rd > filt2nd:
        # g puis r
        cond = (v(mag, mag)[1] - v(mag, mag)[2]) <= 0.3
    else:
        cond = (v(mag, mag)[2] - v(mag, mag)[1]) > 0.0
    return cond

def in_tns(tns_fullname: pd.Series) -> pd.Series: # from fink_filters/rubin/livestream/filter_in_tns/
    """Return alerts with a known counterpart in TNS (AT or confirmed) at the time of emission by Rubin

    Parameters
    ----------
    tns_fullname: pd.Series
        Name according to TNS (string or null).

    Returns
    -------
    out: pd.Series of booleans
        True if in TNS. False otherwise

    Examples
    --------
    >>> s = pd.Series(["SN toto", None, "AT titi"])
    >>> out = in_tns(s)
    >>> assert out.sum() == 2, out.sum()

    >>> from fink_filters.rubin.utils import apply_block
    >>> import pyspark.sql.functions as F
    >>> df = df.withColumn("tns_fullname", F.lit(None).astype("string"))
    >>> df2 = apply_block(df, "fink_filters.rubin.livestream.filter_in_tns.filter.in_tns")
    >>> df2.count()
    0
    """
    in_tns = tns_fullname.apply(lambda x: x is not None)
    return in_tns
    
def extreme_transients(diaSource: pd.DataFrame,
    diaObject: pd.DataFrame,
    simbad_otype: pd.Series,
    mangrove_lum_dist: pd.Series,
    is_sso: pd.Series,
    gaiadr3_DR3Name: pd.Series,
    gaiadr3_Plx: pd.Series,
    gaiadr3_e_Plx: pd.Series,
    vsx_Type: pd.Series,
    legacydr8_zphot: pd.Series,
    firstDiaSourceMjdTaiFink: pd.Series,) -> pd.Series: # keep most of the criteria from /rubin/livestream/filter_extragalactic_lt20mag_candidate
    
    # Loose extragalactic candidate
    f_extragalactic = fb.b_extragalactic_loose_candidate(
        diaSource,
        simbad_otype,
        mangrove_lum_dist,
        is_sso,
        gaiadr3_DR3Name,
        gaiadr3_Plx,
        gaiadr3_e_Plx,
        vsx_Type,
        legacydr8_zphot,
    )  # Xmatch galaxy or Unknown

    # Abs mag at peak
    f_faint = fu.compute_peak_absolute_magnitude(diaSource.psfFlux) < 6
    f_bright = fu.compute_peak_absolute_magnitude(diaSource.psfFlux) >= -16

    f_sampling = (diaObject.nDiaSources > 4) & (
        diaSource.midpointMjdTai - firstDiaSourceMjdTaiFink > 1
    )
    
    # The difference between the g-band and
    # r-band must be almost constant and positive
    condg = lambda mag, filt: mag[~np.isnan(mag) & (filt.astype(int) == 1)]
    condr = lambda mag, filt: mag[~np.isnan(mag) & (filt.astype(int) == 2)]

    meang = np.array([
        np.mean(condg(i, j)) for i, j in zip(cmagpsfc.to_numpy(), cfidc.to_numpy())
    ])
    meanr = np.array([
        np.mean(condr(i, j)) for i, j in zip(cmagpsfc.to_numpy(), cfidc.to_numpy())
    ])
    f_col = (meang - meanr) >= -0.5 & (meang - meanr) < 0.8 # how to calculate g-r at peak?
    
    # Calculate half max flux
    half_flux = 0.5*fu.extract_max_flux(diaSource.psfFlux)
    # How to calculate how many days between data points at half max flux?
    id = np.where(cand["jd"] == half_flux) # check f_range < 10 days
    
    '''How to do this? '''
    # if f_is_rising = fb.b_is_rising(diaSource, diaObject) check if rate is 0.1-1 mag/day
    # if f_is_fading = fb.b_is_fading(diaSource, diaObject) check if rate is 0.3-1 mag/day
    
    f_extreme_transient = (f_faint & f_bright & f_sampling & f_col & f_range & f_is_rising & f_is_fading)
    return f_extreme_transient

if __name__ == "__main__":
    """Test suite for filters"""
    # Run the test suite

    from fink_filters.tester import spark_unit_tests

    globs = globals()
    spark_unit_tests(globs, load_rubin_df=True)
