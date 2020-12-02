[![pypi](https://img.shields.io/pypi/v/fink-filters.svg)](https://pypi.python.org/pypi/fink-filters)

# Fink filters

This repository contains filters used to flag only particular parts of the full stream to be distributed to Fink users. Available filters (i.e. topics) are:

- sn_candidates: alerts identified as SN candidates
- early_sn_candidates: alerts identified as Early SN candidates
- microlensing_candidates: alerts identified as Microlensing event candidates
- sso_ztf_candiates: alerts identified as Solar System Object candidates by ZTF
- sso_fink_candidates: alerts identified as Solar System Object candidates by Fink
- rrlyr: alerts identified as RRLyr in the SIMBAD catalog

## How to contribute?

Learn how to [design](https://fink-broker.readthedocs.io/en/latest/tutorials/create-filters/) your filter, to integrate it inside the Fink broker, and redirect alert streams at your home.

## Installation

If you want to install the package (broker deployment), you can just pip it:

```
pip install fink_filters
```
