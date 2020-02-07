import numpy as np
import pytest
import hypothesis as hyp
import hypothesis.extra.numpy as hypnum

from .. import matcher as mt
from .. import reader as rdr
from .. import permcount as pc
from .. import neighbour as nbr

from ...data import SAMPLE_ZIP, SAMPLE_LONLAT


@pytest.fixture(scope="module")
def nb(cfgcm_test):
    nb = nbr.NeighbourLonLatEuclidean(SAMPLE_LONLAT, 2, [890, 104870])
    nb.find_neighbours()
    return nb


def get_tcs(cfg):
    tcs = rdr.read(SAMPLE_ZIP, cfg=cfg)
    pc.get_ptcs_sttcs(tcs)
    return tcs


@pytest.fixture(scope="module")
def tcs_ref(cfgcm_test):
    """Fixture for all tests that do not append to arguments."""
    return get_tcs(cfgcm_test)


@hyp.given(nulls=hypnum.arrays(bool, 10,
                               elements=hyp.strategies.booleans()),
           weights=hypnum.arrays(float, 10,
                                 elements=hyp.strategies.floats(0.001, 1.),
                                 unique=True))
@hyp.settings(max_examples=30)     
def test_nanaverage(nulls, weights):
    x = np.arange(10, dtype=float)
    x[nulls] = np.nan
    # If every value is NaN, we expect `nanaverage` to raise an error.
    if nulls.sum() == 10:
        with pytest.raises(ZeroDivisionError):
            mt.nanaverage(x, weights=weights)
    else:
        assert (mt.nanaverage(x, weights=weights) ==
                (x[~nulls] * weights[~nulls]).sum() / weights[~nulls].sum())


class TestMatcherRegistrarMatcher:
    """Tests MatcherRegistrar and Matcher."""

    def test_matcherregistrar(self, tcs_ref, nb):

        # Test successful initialization of MatcherBase subclass.
        class MatcherBagheriTest(mt.MatcherBagheri):
            _matcher_type = 'Testing'

        assert mt.MATCHER_REGISTRY['Testing'] is MatcherBagheriTest
        matcher_instance = mt.Matcher('Testing', tcs_ref, nb)
        assert matcher_instance._matcher_type == 'Testing'

        # Pop the dummy class, in case we test twice.
        mt.MatcherRegistrar._registry.pop('Testing', None)

        # Test repeated `_matcher_type` error handling.
        with pytest.raises(ValueError) as excinfo:
            class MatcherBagheriTestBad(mt.MatcherBagheri):
                pass
        assert "already registered in" in str(excinfo.value)

        # Test missing `_matcher_type` error handling.
        with pytest.raises(ValueError) as excinfo:
            class MatcherTestBad2(mt.MatcherBase):
                pass
        assert "must define a" in str(excinfo.value)

    def test_matcher(self, tcs_ref, nb):
        matcher = mt.Matcher('Standard', tcs_ref, nb)
        assert isinstance(matcher, mt.MatcherStandard)
        matcher = mt.Matcher('Bagheri', tcs_ref, nb)
        assert isinstance(matcher, mt.MatcherBagheri)
        with pytest.raises(KeyError):
            matcher = mt.Matcher('Testing')


class TestMatcherBase:

    @pytest.fixture
    def tcs_base(self, cfgcm_test):
        """Fixture for tests in Base that write to tcs.

        This has test scope, so gets re-run each time we run a test.
        """
        return get_tcs(cfgcm_test)

    def test_init(self, tcs_base, nb, cfgcm_test):
        matcher = mt.MatcherBase(tcs_base, nb, cfg=cfgcm_test)
        assert matcher.tcs is tcs_base
        assert matcher.cfg is cfgcm_test
        assert matcher.nb is nb
        assert matcher._average_growth_rate is not None
        assert matcher._disable_tqdm is not cfgcm_test['verbose']

    def test_get_sttc_date_columns(self, tcs_base, nb, cfgcm_test):
        # get_sttc_date_columns is run by __init__.
        matcher_ = mt.MatcherBase(tcs_base, nb, cfg=cfgcm_test)

        for sttc in tcs_base.sttcs.values():
            assert set(['STTC Year', 'Month', 'Day of Week']).issubset(
                set(sttc.data.columns))

    def test_get_backup_ratios_for_nans(self, tcs_base, cfgcm_test):
        # get_backup_ratios_for_nans is run by __init__.
        matcher_ = mt.MatcherBase(tcs_base, nb, cfg=cfgcm_test)

        for ptc in tcs_base.ptcs.values():
            if ptc.ratios['N_avail_days'].isnull().any(axis=None):
                assert set(['DoM_i', 'D_i']).issubset(
                    set(ptc.ratios.keys()))
            else:
                assert 'DoM_i' not in ptc.ratios.keys()
                assert 'D_i' not in ptc.ratios.keys()
