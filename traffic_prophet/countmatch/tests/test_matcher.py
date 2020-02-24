import numpy as np
import pandas as pd
import pytest
import hypothesis as hyp
import hypothesis.extra.numpy as hypnum

from .. import matcher as mt
from .. import reader as rdr
from .. import permcount as pc
from .. import neighbour as nbr

from ...data import SAMPLE_ZIP, SAMPLE_LONLAT


@pytest.fixture(scope='module')
def nb(cfgcm_test):
    nb = nbr.NeighbourLonLatEuclidean(SAMPLE_LONLAT, 2, [890, 104870])
    nb.find_neighbours()
    return nb


def get_tcs(cfg):
    tcs = rdr.read(SAMPLE_ZIP, cfg=cfg)
    pc.get_ptcs_sttcs(tcs)
    return tcs


@pytest.fixture(scope='module')
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

    @pytest.fixture(scope='class')
    def mt_ref(self, nb, cfgcm_test):
        """Fixture for tests on methods that run at initialization."""
        return mt.MatcherBase(get_tcs(cfgcm_test), nb, cfg=cfgcm_test)

    def test_init(self, tcs_base, nb, cfgcm_test):
        matcher = mt.MatcherBase(tcs_base, nb, cfg=cfgcm_test)
        assert matcher.tcs is tcs_base
        assert matcher.cfg is cfgcm_test
        assert matcher.nb is nb
        assert matcher._average_growth_factor is not None
        assert matcher._disable_tqdm is not cfgcm_test['verbose']

    def test_get_sttc_date_columns(self, mt_ref):
        for sttc in mt_ref.tcs.sttcs.values():
            assert set(['STTC Year', 'Month', 'Day of Week']).issubset(
                set(sttc.data.columns))

    def test_get_backup_ratios_for_nans(self, mt_ref):
        """Test `get_backup_ratios_for_nans`.

        Implicitly tests `get_annually_averaged_ratios`.
        """
        for ptc in mt_ref.tcs.ptcs.values():
            if ptc.ratios['N_avail_days'].isnull().any(axis=None):
                for year in ptc.perm_years:
                    assert (ptc.ratios['DoM_i'].at[year, 'DoM_i'] ==
                            mt.nanaverage(
                                ptc.ratios['DoM_ijd'].loc[year].values,
                                ptc.ratios['N_avail_days'].loc[year].values))
                    # Currently using medians rather than means.
                    assert (ptc.ratios['D_i'].at[year, 'D_i'] ==
                            ptc.ratios['D_ijd'].loc[year].unstack().median())
            else:
                assert not (set(['DoM_i', 'D_i', 'avail_years'])
                            .intersection(set(ptc.ratios.keys())))

    @staticmethod
    def get_available_years(df_N):
        """Create a dataframe of permanent years where ratios are available."""
        # Old version from countmatch.matcher.
        avail_years = []
        month = []
        for name, group in (df_N.notnull().groupby('Month')):
            gd = group.reset_index(level='Month', drop=True)
            avail_years.append([list(gd.loc[gd[c]].index.values)
                                for c in group.columns])
            month.append(name)
        return pd.DataFrame(avail_years, index=month)

    @hyp.given(n_nan=hyp.strategies.integers(min_value=5, max_value=20))
    @hyp.settings(max_examples=30)
    def test_get_available_years(self, tcs_ref, n_nan):
        """Fuzz test for `get_available_years`."""
        # -104870 is the only SAMPLE_ZIP PTC with >1 yr of perm count data.
        df = tcs_ref.ptcs[-104870].ratios['N_avail_days']

        # Randomly drop available days.
        arr = df.values.flatten()
        nan_indices = np.random.choice(np.arange(arr.size),
                                       size=n_nan, replace=False)
        arr[nan_indices] = np.nan
        df_N = pd.DataFrame(arr.reshape(df.shape), index=df.index,
                            columns=df.columns)

        # Curiously, this breaks if the elements of avail_years are arrays
        # rather than lists; in the latter case, we'd have to loop over
        # elements.
        assert self.get_available_years(df_N).equals(
            mt.MatcherBase.get_available_years(df_N))

    def test_get_average_growth(self, mt_ref):
        assert mt_ref._average_growth_factor == np.mean([
            mt_ref.tcs.ptcs[-890].growth_factor,
            mt_ref.tcs.ptcs[-104870].growth_factor])

        assert (mt_ref.get_average_growth(multi_year=True) ==
                mt_ref.tcs.ptcs[-104870].growth_factor)

    def test_get_neighbour_ptcs(self, tcs_base, nb, cfgcm_test):
        # This is fundamentally just a lookup method for `nb.get_neighbours`.
        # We only have two PTCs (for the negative direction), so we can only
        # check those.

        # First, check default - 2 neighbours.
        matcher = mt.MatcherBase(tcs_base, nb, cfg=cfgcm_test.copy())
        neighbours = matcher.get_neighbour_ptcs(matcher.tcs.sttcs[-241])
        one_dir_neighbours = [p.count_id for p in neighbours]
        assert np.array_equal(one_dir_neighbours,
                              -1 * np.array(nb.get_neighbours(241)[0]))

        # Check getting any-direction neighbours.  There are no
        # positive-direction neighbours, so this should return as above.
        matcher.cfg['match_single_direction'] = False
        neighbours = matcher.get_neighbour_ptcs(matcher.tcs.sttcs[-241])
        assert np.array_equal([p.count_id for p in neighbours],
                              one_dir_neighbours)

        # Now check retrieving only one neighbour.
        matcher.cfg['n_neighbours'] = 1
        matcher.cfg['match_single_direction'] = True
        neighbours = matcher.get_neighbour_ptcs(matcher.tcs.sttcs[-446378])
        assert np.array_equal([p.count_id for p in neighbours],
                              [-1 * nb.get_neighbours(446378)[0][0]])

        # Check that getting same-direction neighbours for a positive-direction
        # STTC fails.
        with pytest.raises(ValueError) as excinfo:
            neighbours = matcher.get_neighbour_ptcs(matcher.tcs.sttcs[170])
        assert "too few available PTC locations" in str(excinfo.value)

        # Getting any-direction neighbours should succeed.
        matcher.cfg['match_single_direction'] = False
        neighbours = matcher.get_neighbour_ptcs(matcher.tcs.sttcs[170])
        assert np.array_equal([p.count_id for p in neighbours],
                              [-1 * nb.get_neighbours(170)[0][0]])

    @pytest.mark.parametrize(
        ['sy', 'py', 'closest'],
        [(2006, np.array([2011, 2012, 2013]), 2011),
         (np.array([2010, 2011, 2012]),
          np.array([2011, 2012, 2013]), np.array([2011, 2011, 2012])),
         (np.array([2012, 2019]), np.array([2011, 2013, 2017]),
          np.array([2011, 2017]))])
    def test_get_closest_year(self, sy, py, closest):
        """Fuzz test for `get_closest_year`."""
        if isinstance(sy, np.ndarray):
            assert np.array_equal(
                mt.MatcherBase.get_closest_year(sy, py), closest)
        else:
            assert mt.MatcherBase.get_closest_year(sy, py) == closest

    @hyp.given(idx=hyp.strategies.integers(min_value=0, max_value=23),
               col=hyp.strategies.integers(min_value=0, max_value=6))
    @hyp.settings(max_examples=30)
    def test_ratio_lookup_regular(self, mt_ref, idx, col):
        """Fuzz test for `ratio_lookup` when default time is available."""
        ptc = mt_ref.tcs.ptcs[-104870]
        sttc_year, sttc_month = ptc.ratios['D_ijd'].index[idx]
        sttc_dow = ptc.ratios['D_ijd'].columns[col]

        sttc_row = pd.Series([sttc_year, sttc_month, sttc_dow],
                             index=['STTC Year', 'Month', 'Day of Week'])
        default_closest_years = dict(zip(
            [2010, 2012],
            mt_ref.get_closest_year(np.array([2010, 2012]), ptc.perm_years)))

        # Set idx, col to correct lookup for the lone NaN in -104870.
        if (idx, col) == (4, 4):
            idx = 16
            sttc_year = 2012
        output_ref = (sttc_year, ptc.ratios['DoM_ijd'].iat[idx, col],
                      ptc.ratios['D_ijd'].iat[idx, col])

        assert (output_ref ==
                mt_ref.ratio_lookup(sttc_row, ptc, default_closest_years))

    def test_ratio_lookup_exceptional(self, tcs_base, nb, cfgcm_test):
        """Test for `ratio_lookup` when default time is not available."""
        matcher = mt.MatcherBase(tcs_base, nb, cfg=cfgcm_test)
        ptc = matcher.tcs.ptcs[-104870]
        default_closest_years = dict(zip(
            [2010, 2012],
            matcher.get_closest_year(np.array([2010, 2012]), ptc.perm_years)))

        # Selectively remove some values.
        ptc.ratios['DoM_ijd'].iat[3, 1] = np.nan
        ptc.ratios['DoM_ijd'].iat[15, 1] = np.nan
        ptc.ratios['DoM_ijd'].iat[6, 4] = np.nan
        ptc.ratios['DoM_ijd'].iat[18, 4] = np.nan
        ptc.ratios['avail_years'].iat[3, 1] = []
        ptc.ratios['avail_years'].iat[6, 4] = []

        # For the NaN at 4, 4, check that we get the 2012 value.
        row_idx = ['STTC Year', 'Month', 'Day of Week']
        sttc_row = pd.Series([2010, 5, 4], index=row_idx)
        output_ref = (2012, ptc.ratios['DoM_ijd'].iat[16, 4],
                      ptc.ratios['D_ijd'].iat[16, 4])
        assert (output_ref ==
                matcher.ratio_lookup(sttc_row, ptc, default_closest_years))

        # For the removed values, check that we get the annual average instead.
        sttc_row = pd.Series([2010, 4, 1], index=row_idx)
        output_ref = (2010, ptc.ratios['DoM_i'].at[2010, 'DoM_i'],
                      ptc.ratios['D_i'].at[2010, 'D_i'])
        assert (output_ref ==
                matcher.ratio_lookup(sttc_row, ptc, default_closest_years))

        sttc_row = pd.Series([2012, 7, 4], index=row_idx)
        output_ref = (2012, ptc.ratios['DoM_i'].at[2012, 'DoM_i'],
                      ptc.ratios['D_i'].at[2012, 'D_i'])
        assert (output_ref ==
                matcher.ratio_lookup(sttc_row, ptc, default_closest_years))

    def test_get_ratio_from_ptc(self, tcs_base, nb, cfgcm_test):
        """Test for `get_ratio_from_ptc`."""
        matcher = mt.MatcherBase(tcs_base, nb, cfg=cfgcm_test)
        ptc = matcher.tcs.ptcs[-104870]
        default_closest_years = dict(
            zip([2010, 2011, 2012], matcher.get_closest_year(
                np.array([2010, 2011, 2012]),ptc.perm_years)))

        # Selectively remove some values.
        ptc.ratios['DoM_ijd'].iat[3, 1] = np.nan
        ptc.ratios['DoM_ijd'].iat[15, 1] = np.nan
        ptc.ratios['avail_years'].iat[3, 1] = []

        # Check that function outputs are identical to those of `ratio_lookup`,
        # line by line.
        for key in (-241, 170, -1978):
            sttc = matcher.tcs.sttcs[key]
            # Try both direct matching and merging
            ratios = matcher.get_ratio_from_ptc(sttc, ptc,
                                                n_switch_to_merge=9999)
            ratios_merged = matcher.get_ratio_from_ptc(sttc, ptc,
                                                       n_switch_to_merge=0)

            for idx, row in ratios.iterrows():
                # row's column names are same as STTC row inputs, except as
                # floats due to iterrows typecasting.
                sttc_row = pd.Series(
                    [int(row['STTC Year']), int(row['Month']),
                     int(row['Day of Week'])],
                    index=['STTC Year', 'Month', 'Day of Week'])
                output_ref = matcher.ratio_lookup(
                    sttc_row, ptc, default_closest_years)
                assert (output_ref == (int(row['Closest PTC Year']),
                                       row['DoM_ijd'], row['D_ijd']))

                # # Check if row is same regardless if it is made through
                # # merging.
                assert row.equals(ratios_merged.loc[idx, :])
