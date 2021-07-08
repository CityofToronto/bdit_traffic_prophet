# Traffic Prophet

Traffic Prophet is a suite of Python tools to transform Toronto's historical
traffic count data to estimates of annual average daily traffic (AADT) on all
city streets. It is based on the Traffic Emissions Prediction Scheme (TEPS) codebase
created by Dr. Arman Ganji of the University of Toronto ([paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12508)).

Traffic Prophet currently remains under development.  On initial release, it
will contain:

* A method, based on [Bagheri et al. 2014](
https://ascelibrary.org/doi/abs/10.1061/%28ASCE%29TE.1943-5436.0000528), of
extrapolating short-term traffic counts into AADTs using nearby long-term
counts.
* Gaussian process regression method for estimating arterial AADTs from nearby
observed AADTs.
* Support vector regression method for estimating local road AADTs from nearby
observed AADTs.

Alongside the Traffic Prophet codebase, this repo also contains some literature
review, a number of development and experimental notebooks in a `sandbox`
folder, utilities for working with TEPS and scripts for helping to generate
input data for the model.

## Folder Structure

- `input_data` - scripts to produce input data tables on Postgres.
- `lit_review` - literature review, notably for Bagheri et al. 2013, the basis
  for CountMatch.
- `sandbox` - Developmental and experimental notebooks in support of building
  out TEPS.
- `teps` - tutorials on how to run TEPS, and how to bootstrap components of
  Traffic Prophet to produce input datasets for TEPS.
- `traffic_prophet` - Traffic Prophet codebase. See below for details.

### Traffic Prophet Codebase

- `connection.py` - basic wrapper for storing Postgres connections.
- `config_template.py` - template for creating `config.py`.
- `countmatch/` - port of the TEPS version of the Bagheri et al. 2014 algorithm.
  - `base.py` - base classes and shared code.
  - `derivedvals.py` - aggregate values and weights, such as MADT, and `DoM_ijd`
    (ratio between MADT and day-to-month ADT).
  - `growthfactor.py` - fit for year-on-year multiplicative growth factor.
  - `matcher.py` - matcher between short and permanent counts, and AADT
    estimator classes.
  - `neighbour.py` - nearest neighbour calculator.
  - `permcount.py` - find which counts in a dataset have enough data to be
    considered permanent counts.
  - `reader.py` - reads
  - `tests/` - test suite for `countmatch`
    - `conftest.py` - common components of tests.
    - `test_{countmatch_file}.py` - test suite for {countmatch_file}.
- `data/` - sample data used in test suites. See `__init__.py` for definitions.
- `tests/` - integration testing for the entire volume model. Currently empty.

## Requirements

As given by `requirements.txt`, the Traffic Prophet codebase requires:

```
hypothesis>=5.5.4
numpy>=1.18
pandas>=1.0
psycopg2>=2.8.4
pytest>=5.4
scikit-learn>=0.22
statsmodels>=0.11.1
tqdm>=4.43
```

Individual Jupyter notebooks in `sandbox` may require other packages (including
`notebook` to actually run Jupyter notebooks, of course).

## Usage

### Installation

`traffic_prophet` itself only requires a Python environment with the packages
listed above. It, however, also relies on input data from Postgres (or zip
files, but this is a legacy ingestion method developed for testing against
TEPS). The required Postgres tables are:

```
prj_volume.tp_centreline_lonlat.sql
prj_volume.tp_centreline_volumes.sql
prj_volume.tp_daily_volumes.sql
```

The scripts to create these are located in `input_data/flow`.

### Importing

To import `traffic_prophet`, add this folder to the Python PATH, eg. with

```
import sys
sys.path.append('<FULL PATH OF bdit_traffic_prophet>')
```

### Testing

To test the package, run the following in this folder:

```
pytest -s -v --pyargs traffic_prophet
```

## License

Traffic Prophet is licensed under the GNU General Public License v3.0 - see the
`LICENSE` file.
