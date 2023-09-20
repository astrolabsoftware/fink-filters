from fink_filters.classification import extract_fink_classification_
from fink_utils.xmatch.simbad import return_list_of_eg_host

from astropy.coordinates import SkyCoord


def gvom_filter(
    rb,
    magpsf,
    ra,
    dec,
    cdsxmatch,
    roid,
    mulens,
    snn_snia_vs_nonia,
    snn_sn_vs_all,
    rf_snia_vs_nonia,
    ndethist,
    drb,
    classtar,
    jd,
    jdstarthist,
    rf_kn_vs_nonkn,
    tracklet,
):
    classification = extract_fink_classification_(
        cdsxmatch,
        roid,
        mulens,
        snn_snia_vs_nonia,
        snn_sn_vs_all,
        rf_snia_vs_nonia,
        ndethist,
        drb,
        classtar,
        jd,
        jdstarthist,
        rf_kn_vs_nonkn,
        tracklet,
    )

    # non-buggy alerts
    f_bogus = rb >= 0.9

    # extra-galactic alerts
    base_extragalactic = return_list_of_eg_host()
    fink_extragalactic = [
        "KN candidate",
        "SN candidate",
        "Early SN Ia candidate",
        "Ambiguous",
    ]
    extragalactic = list(base_extragalactic) + list(fink_extragalactic)
    f_class = classification.isin(extragalactic)

    coord = SkyCoord(ra, dec, unit="deg")
    # alerts not in the milky way
    gal_latitude = coord.galactic.b
    mask_south_gal = gal_latitude < -15
    mask_north_gal = gal_latitude > 15
    f_gal = mask_north_gal & mask_south_gal

    # alerts not in the ecliptic
    ecl_latitude = coord.transform_to("geocentricmeanecliptic").lat
    mask_south_ecl = ecl_latitude < -15
    mask_north_ecl = ecl_latitude > 15
    f_ecl = mask_north_ecl & mask_south_ecl

    # bright alerts
    f_brightness = magpsf <= 17.0

    # short living transient
    f_short = jd - jdstarthist <= 5

    f_gvom = f_bogus & f_class & f_short & f_brightness & f_gal & f_ecl
    return f_gvom
