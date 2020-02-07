"""CountMatch test suite preprocessing.

Fixtures used by multiple files in this and subdirectories are placed here.
See https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions
for more.

"""

import pytest

from .. import reader

from ...data import SAMPLE_ZIP


@pytest.fixture(scope="session")
def cfgcm_test():
    # Configuration settings for tests.
    return {
        'verbose': False,
        'min_count': 96,
        'min_counts_in_day': 24,
        'min_permanent_months': 12,
        'min_permanent_days': 274,
        'exclude_ptc_neg': [446378, ],
        'exclude_ptc_pos': [],
        'derived_vals_calculator': 'Standard',
        'derived_vals_settings': {},
        'growth_factor_calculator': 'Composite',
        'growth_factor_settings': {},
        'min_year': 2006,
        'max_year': 2018,
        'average_growth': True
    }


@pytest.fixture(scope="session")
def sample_counts(cfgcm_test):
    return reader.read(SAMPLE_ZIP, cfg=cfgcm_test)
