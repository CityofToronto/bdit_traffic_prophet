import pytest
import hypothesis as hyp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .. import reader
from .. import permcount as pc
from .. import derivedvals as dv
from .. import growthfactor as gf

from ...data import SAMPLE_ZIP


def get_single_ptc(sample_counts, cfgcm_test, count_id):
    pcpp = pc.PermCountProcessor(None, None, cfg=cfgcm_test)
    perm_years = pcpp.partition_years(sample_counts.counts[count_id])
    ptc = pc.PermCount.from_count_object(sample_counts.counts[count_id],
                                         perm_years)
    dvs = dv.DerivedVals('Standard')
    dvs.get_derived_vals(ptc)
    return ptc


@pytest.fixture(scope="module")
def ptc_oneyear(sample_counts, cfgcm_test):
    return get_single_ptc(sample_counts, cfgcm_test, -890)


@pytest.fixture(scope="module")
def ptc_multiyear(sample_counts, cfgcm_test):
    return get_single_ptc(sample_counts, cfgcm_test, -104870)


class TestGrowthFactorBase:
    """Test growth factor base class."""

    def setup(self):
        self.gfb = gf.GrowthFactorBase()

    def test_get_aadt(self, ptc_oneyear, ptc_multiyear):
        for ptc in (ptc_oneyear, ptc_multiyear):
            aadt = self.gfb.get_aadt(ptc)
            assert list(np.sort(aadt.columns)) == ['AADT', 'Year']
            assert aadt['Year'].dtype == np.dtype('float64')
            assert np.array_equal(ptc.data['AADT'].index.values,
                                  aadt['Year'].values)
            assert np.array_equal(ptc.data['AADT']['AADT'].values,
                                  aadt['AADT'].values)

    def test_get_wadt_py(self, ptc_oneyear, ptc_multiyear):
        # For single year PTC, confirm WADT values for individual weeks.
        wadt_oy = self.gfb.get_wadt_py(ptc_oneyear)
        wadt_jun14 = (ptc_oneyear.data['Daily Count']
                      .loc[(2010, 165):(2010, 171), 'Daily Count'].mean())
        wadt_nov29 = (ptc_oneyear.data['Daily Count']
                      .loc[(2010, 333):(2010, 339), 'Daily Count'].mean())
        assert np.isclose(
            (wadt_oy.loc[wadt_oy['Start of Week'] == '2010-06-14', 'WADT']
             .values[0]), wadt_jun14)
        assert np.isclose(
            (wadt_oy.loc[wadt_oy['Start of Week'] == '2010-11-29', 'WADT']
             .values[0]), wadt_nov29)

        # For multiyear PTC, confirm we can reproduce data frame.
        wadt_my = self.gfb.get_wadt_py(ptc_multiyear)

        ptc_multiyear_dc = ptc_multiyear.data['Daily Count'].loc[
            ptc_multiyear.perm_years, :].copy()
        ptc_multiyear_dc['Week'] = (
            ptc_multiyear_dc['Date'] -
            (ptc_multiyear_dc['Date'].dt.dayofweek *
             np.timedelta64(1, 'D'))).dt.week

        wadt_ref = (ptc_multiyear_dc.loc[ptc_multiyear_dc['Week'] < 53, :]
                    .groupby(['Year', 'Week']).agg(['mean', 'count']))
        wadt_ref.columns = ('WADT', 'N_days')
        wadt_ref = wadt_ref.loc[wadt_ref['N_days'] == 7, :].reset_index()

        wadt_ref['Week'] = (wadt_ref['Week'].astype(float) +
                            52. * (wadt_ref['Year'] - wadt_ref['Year'].min()))

        # Check that all weeks with 7 days are in wadt.
        assert np.array_equal(wadt_ref['Week'], wadt_my['Week'])
        assert np.array_equal(wadt_ref['WADT'], wadt_my['WADT'])
