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
"""Select alerts with a known counterpart in TNS at the time of emission by Rubin"""

import pandas as pd

DESCRIPTION = (
    "Select alerts with a known counterpart in TNS at the time of emission by Rubin"
)


def in_tns(tns_type: pd.Series) -> pd.Series:
    """Return alerts with a known counterpart in TNS at the time of emission by Rubin

    Parameters
    ----------
    tns_type: pd.Series
        Type according to TNS (string or null).

    Returns
    -------
    out: pd.Series of booleans
        True if in TNS. False otherwise

    Examples
    --------
    >>> s = pd.Series(["SN Ia", None, None])
    >>> out = in_tns(s)
    >>> out.sum() == 1
    """
    in_tns = tns_type.apply(lambda x: x is not None)
    return in_tns
