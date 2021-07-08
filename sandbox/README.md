# Volume Model Sandbox

This is a sandbox for experimenting with new technologies to use for the Volume
Model. Each experiment is in a folder, whose name follows the format

```
<DATE>_<EXPERIMENT_TYPE>
```

Over time, all experimental development of Traffic Prophet was archived in this
folder as well. Notably, the bulk of the CountMatch algorithm development is in
`20191222_countmatch_prototype/`.

- `20190919_intermediate_file_io/` - performance testing for reading TEPS zip
  files.
- `20191009_timestamp_speed_test/` - performance testing for various ways to
  decode timestamps in Python.
- `20191120_read_from_postgres/` - smoke test for `reader`'s Postgres IO
  capability. Note that `czhu.btp_centreline_lonlat` is now
  `prj_volume.tp_centreline_lonlat`, and `czhu.btp_centreline_daily_counts` is
  now `prj_volume.tp_centreline_daily_counts`.
- `20191209_prep_kriglocal/` - notebooks that try to replicate how TEPS produces
  datasets for KCOUNT and LocalSVR. Can be used to produce input datasets for
  Traffic Prophet equivalents.
- `20191222_countmatch_prototype/` - prototype to reproduce TEPS's
  interpretation of Bagheri et al. 2014.
  - `CountmatchDev1-Prototype.ipynb` - initial prototype.
  - `CountmatchDev2-ReproducingArmanMAE.ipynb` - checks TEPS and Traffic Prophet
    predictions against ground truth, using a port of TEPS's validation scheme.
    Concluded that TEPS's validation scheme is flawed.
  - `CountmatchDev3-SensibleMatcherPrototype.ipynb` - refactoring of initial
    prototype.
  - `countmatch_bagheri.py` - CountMatch prototype, based directly on Bagheri et
    al. 2014.
  - `countmatch_common.py` - common functions between the different flavours of
    CountMatch prototype.
  - `countmatch_teps.py` - CountMatch prototype, based on TEPS.
  - `countmatch_hybrid.py` - CountMatch prototype, combining best practices
    between TEPS and Bagheri.
  - `CountmatchDev4-FunctionTests.ipynb` - tests to check that
    `countmatch_bagheri.py`, `countmatch_hybrid.py`, and `countmatch_teps.py`
    can be run.
  - `CountmatchDev5-FakeDataGenerator.ipynb` - generates fake short-term counts from
    permanent counts, so that we can test the relative predictive accuracies of
    the three versions of CountMatch.
  - `countmatch_validator.py` - fake data generator and testing framework to
    determine relative performance of different flavours of CountMatch
    prototype.
  - `CountmatchDev6-Shootout.ipynb` - 3-way relative comparison between
    CountMatch prototypes, informing final design of CountMatch module.
  - `MatcherPlotter.ipynb` - routines for visualizing centreline network.
- `20200119_countmatch_refactor_check/` - test CountMatch code refactoring.
- `20200120_countmatch_imputer/` - test imputer for filling in NaNs in `DoM_ijd`
  and `D_ijd` matrixes.
- `20200204_zips_for_teps/` - dump 2017-2018 data to produce updated numbers
  for EED's 2018 GHG inventory.
- `20200218-tepsrerunwithnewdata/` - analyze results from running TEPS with 2018
  and 2019 data.
- `20200315-countmatchintegrationtests/` - integration testing of CountMatch.
- `20200724_zips_for_arman/` - dump 2016-2018 data for Arman.
- `20201216_teps2019/` - dump 2019, and revised 2017-2018 data, for EED's 2019
  GHG inventory.
