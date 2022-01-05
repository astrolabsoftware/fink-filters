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

import numpy as np
import pandas as pd
import h5py


def make_mangrove_pdf(path_in, path_out='mangrove_filtered.csv',
                      range_interferometers=230):
    """
    Create a pandas dataframe needed in early_kn_candidates from the hdf5
    Mangrove catalog and save it as csv.

    early_kn_candidate loads the pre-filtered csv file, the aim of this
    function is to create a new csv file if the catalog is updated or the range
    change.

    The index of the galaxy and the stellar mass are not needed in the
    early_kn_candidates function so they may be removed. They are here to be
    used for information purposes only.

    Parameters
    ----------
    path_in : string
        path to Mangrove hdf5 catalog.
    path_out : string, optional
        path where the csv file will be saved. Default: 'mangrove_filtered.csv'
    range_interferometers: float
        range of the interferometers in Mpc. Only the galaxies in this range
        will be considered

    Returns
    -------
    None

    """
    pdf_mangrove = pd.DataFrame(np.array(
        h5py.File(path_in, 'r')['__astropy_table__']
    ))
    pdf_mangrove = pdf_mangrove.loc[:, [
        'HyperLEDA_name',    # Name in the HyperLEDA catalog
        '2MASS_name',    # Name in the 2MASS XSC catalog
        'RA',           # Right ascention [deg] of the GLADE galaxy
        'dec',          # Declination [deg] of the GLADE galaxy
        'dist',         # Luminosity distance [Mpc] of the GLADE galaxy,
        'dist_err',     # Error of distance [Mpc] of the GLADE galaxy
        'z',            # Redshift of the GLADE galaxy
        'stellarmass',  # Determined stellar mass of the the associated AllWISE
                        # object, empty if no association
    ]]
    pdf_mangrove = pdf_mangrove[pdf_mangrove.dist < range_interferometers]
    pdf_mangrove['ang_dist'] = pdf_mangrove.dist / np.square(1 + pdf_mangrove.z)
    pdf_mangrove.drop(columns=['z'], inplace=True)
    pdf_mangrove.rename(columns={'idx': 'galaxy_idx', 'RA': 'ra',
                                 'dist': 'lum_dist'}, inplace=True)
    pdf_mangrove.sort_values(by='lum_dist', inplace=True)
    pdf_mangrove.to_csv(path_out, index=False)
