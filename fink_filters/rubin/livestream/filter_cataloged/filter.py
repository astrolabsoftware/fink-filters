# Copyright 2019-2026 AstroLab Software
# Author: Julien Peloton
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
"""Return LSST alerts with matches in catalogs"""

import pandas as pd

DESCRIPTION = (
    "Select alerts with a counterpart within 1.5 arcseconds in SIMBAD or GAIA DR3"
)


def cataloged(is_cataloged: pd.Series) -> pd.Series:
    """Flag for cataloged alerts in Rubin

    Parameters
    ----------
    is_cataloged: pd.Series
        Series containing booleans from `pred.is_cataloged`
        in alert packets.

    Returns
    -------
    out: pd.Series
        Booleans: True for alerts with a match in catalogs,
        False otherwise.

    Examples
    --------
    >>> s = pd.Series([True, False, True])
    >>> out = cataloged(s)
    >>> out.sum() == 2
    """
    # FIXME: this is super dummy for the moment
    # this is a test
    return is_cataloged
