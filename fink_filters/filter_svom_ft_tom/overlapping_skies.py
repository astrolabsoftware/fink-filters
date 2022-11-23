#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:37:49 2022

@author: dt270490
"""

import numpy as np
import healpy as hp
from astroplan import Observer, FixedTarget
from astroplan import is_observable, is_always_observable, months_observable
from astroplan import AltitudeConstraint, AirmassConstraint, AtNightConstraint
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from mhealpy import HealpixMap
import mhealpy as hmap
import pandas as pd
import json


def load_observatories(obs_path_file: str):
    """
    Load the observatories for which we want to crossmacth the observable 
    skies

    Parameters
    ----------
    obs_path_file : str
        Path to the files where the observatory settings are stored.

    Returns
    -------
    observatories : Dataframes
        pandas Dataframe of the observatory configurations 
        to feed the astroplan.Observer package.

    """
    observatories = pd.read_csv(obs_path_file, delimiter=";")
    return observatories


def simu_night_time_interval(ref_obs, ref_date: str, n_days: int, day_bin: int):
    """
    Compute the list of the date interval during which the observable skies
    will be crossmatched

    Parameters
    ----------
    ref_obs : astroplan.observer.Observer
        astroplan.observer.Observer object for the Observatory chosen as the 
        reference observatory.
    ref_date : str
        Date of reference to start the simulation.
    n_days : int
        Number of simulated days.
    day_bin : int
        Day interval between two simulated nights.

    Returns
    -------
    None.

    """
    # Initialize the time ranges by starting 1h after astro. twiligtht at VRO
    date_0 = Time(ref_date)
    time_interval = np.arange(0, n_days * day_bin, day_bin)

    dates_start = date_0 + time_interval * u.day
    dates_night = ref_obs.tonight(horizon=-13 * u.degree)
    if dates_night[0] < date_0:
        dates_night = ref_obs.tonight(date_0 + 0 * u.day, horizon=-13 * u.degree)
    # dates_end_range = [dates_night[0][0]+2*u.hr]
    time_ranges = [Time([dates_night[0].iso, dates_night[1].iso])]
    # dates_start_night = ref_obs.twilight_evening_astronomical(date_0,
    #                                                             which='nearest')
    # dates_end_range = dates_start_night+24*u.hr
    # time_ranges = [Time([dates_start_night.iso, dates_end_range.iso])]
    return time_ranges


def build_targets(ras: np.array, decs: np.array):
    """
    Make the target objects to be ingested by the astroplan is_observable
    functions

    Parameters
    ----------
    ras : np.array
        Numpy arrays of right ascencion in units of degrees.
    decs : np.array
        Numpy arrays of declinations in units of degrees.

    Returns
    -------
    targets : 
        List of targets for which we want to estimate the observability

    """
    ras_grid, decs_grid = np.meshgrid(ras, decs)

    target_table = Table()
    target_table["ra"] = np.reshape(ras_grid, ras_grid.shape[0] * ras_grid.shape[1])
    target_table["dec"] = np.reshape(decs_grid, decs_grid.shape[0] * decs_grid.shape[1])

    targets = FixedTarget(
        coord=SkyCoord(ra=target_table["ra"] * u.deg, dec=target_table["dec"] * u.deg)
    )

    return targets, target_table


def make_visibility_masks(constraints, observatory, targets, time_range: list):
    """
    Build the skymap mask regarding the obsverational constraints:  
    True = the position is visible
    False = the position is not visible

    Parameters
    ----------
    constraints : TYPE
        DESCRIPTION.
    observatory : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    time_range : list
        time range during which the observability is estimated

    Returns
    -------
    mask_visibility : np.array
        Numpy arrays of booleans

    """

    mask_visibility = is_observable(
        constraints, observatory, targets, time_range=time_range
    )

    return mask_visibility


def plot_common_sky(
    nside: int,
    scheme: str,
    target_table: Table,
    mask_common_sky: np.array,
    title_sky: str,
):

    is_nested = scheme == "nested"
    m1 = HealpixMap(nside=nside, scheme=scheme, density=True)
    m1 = np.zeros(hp.nside2npix(nside))

    theta = np.deg2rad((target_table[mask_common_sky]["dec"] * -1) + 90)
    phi = np.deg2rad(target_table[mask_common_sky]["ra"])

    sample_pix = hp.ang2pix(nside, theta, phi, nest=is_nested)

    m1[sample_pix] = 1
    # fig = plt.figure()
    # ax = fig.add_axes((0,0,1,1))
    # m1.plot(ax)
    hp.mollview(m1, nest=is_nested, title=title_sky)


if __name__ == "__main__":
    obs_path_file = "/local/home/dt270490/Documents/FINK/"
    obs_filename = "svom_network.csv"
    observatories = load_observatories(obs_path_file + obs_filename)

    # Take VRO Cerro Pachon as the reference observatory
    ref_obs_name = "Cerro Pachon"
    vro = Observer.at_site(ref_obs_name)
    # Setting up parameters
    ref_date = vro.tonight(horizon=-13 * u.degree)
    time_ranges_vro = Time([ref_date[0].iso, (ref_date[0] + 30 * u.second).iso])
    n_days = 1
    day_bin = 1
    # Define the grid
    nside = 32
    scheme = "nested"
    ras = np.linspace(0, 360, 181)
    decs = np.linspace(-90, 90, 181)

    # Build the targets
    targets, target_table = build_targets(ras, decs)

    constraints = [
        AltitudeConstraint(40 * u.deg, 80 * u.deg),
        AirmassConstraint(2),
    ]  # , AtNightConstraint.twilight_astronomical()]
    # Loop over the Observatory we want to test
    masks_observable_vro = []
    masks_observable_colibri = []
    masks_both_observability = []
    sky_frac_visibilities = []
    tstarts = []
    tends = []
    for i in range(len(observatories)):
        try:
            observatory = Observer.at_site(observatories.obs_name[i])
        except:
            observatory = Observer(
                longitude=observatories.longitude[i] * u.deg,
                latitude=observatories.latitude[i] * u.deg,
                elevation=observatories.elevation[i] * u.m,
                name=observatories.obs_name[i],
            )

        # Build the time ranges
        time_ranges = simu_night_time_interval(
            observatory, ref_date[0], n_days, day_bin
        )
        tstarts.append(time_ranges[0][0].isot)
        tends.append(time_ranges[0][1].isot)
        for time_range in time_ranges:
            mask_visibility_vro = make_visibility_masks(
                constraints, vro, targets, time_ranges_vro
            )
            mask_visibility_observatory = make_visibility_masks(
                constraints, observatory, targets, time_range
            )
            mask_both_observability = mask_visibility_vro & mask_visibility_observatory
            sky_frac_visibility = (
                mask_both_observability.sum() / mask_visibility_vro.sum()
            )
            masks_observable_vro.append(mask_visibility_vro)
            masks_observable_colibri.append(mask_visibility_observatory)
            masks_both_observability.append(mask_both_observability)
            sky_frac_visibilities.append(sky_frac_visibility)

            title_combi = (
                "VRO-"
                + observatories.obs_name[i]
                + " overlapping skies = "
                + str("%.2f" % (sky_frac_visibility * 100))
                + " %"
            )
            print(title_combi)

            plot_common_sky(
                nside, scheme, target_table, mask_both_observability, title_combi
            )

    res = {}
    res["tstart"] = tstarts
    res["tend"] = tends
    res["observatory_names"] = observatories.obs_name.values.tolist()
    res["sky_frac_visibilities"] = sky_frac_visibilities

    json_object = json.dumps(res, indent=4)
    with open(obs_path_file + "overlap_skies.json", "w") as outfile:
        outfile.write(json_object)
