import pytest
import hypothesis as hyp
import numpy as np
import pandas as pd

from .. import permcount as pc
from .. import derivedvals as dv


def get_single_ptc(sample_counts, cfgcm_test, count_id):
    pcpp = pc.PermCountProcessor(None, None, cfg=cfgcm_test)
    perm_years = pcpp.partition_years(sample_counts.counts[count_id])
    ptc = pc.PermCount.from_count_object(sample_counts.counts[count_id],
                                         perm_years)
    return ptc


@pytest.fixture(scope='module')
def ptc_oneyear(sample_counts, cfgcm_test):
    return get_single_ptc(sample_counts, cfgcm_test, -890)


@pytest.fixture(scope='module')
def ptc_multiyear(sample_counts, cfgcm_test):
    return get_single_ptc(sample_counts, cfgcm_test, -104870)


class TestDerivedValsBase:

    def setup(self):
        self.dvc = dv.DerivedValsBase()

    def test_preprocess_daily_counts(self, ptc_multiyear):
        dca = self.dvc.preprocess_daily_counts(ptc_multiyear.data)
        assert 'Month' in dca.columns
        assert 'Day of Week' in dca.columns

    def test_get_madt(self, ptc_oneyear, ptc_multiyear):
        for ptc in (ptc_oneyear, ptc_multiyear):
            dca = self.dvc.preprocess_daily_counts(ptc.data)
            madt = self.dvc.get_madt(dca)

            madt_ref = pd.DataFrame({
                'MADT': dca.groupby(['Year', 'Month'])['Daily Count'].mean(),
                'Days Available': dca.groupby(
                    ['Year', 'Month'])['Daily Count'].count()},
                index=pd.MultiIndex.from_product(
                    [dca.index.levels[0], np.arange(1, 13, dtype=int)],
                    names=['Year', 'Month']))
            madt_ref['Days in Month'] = [
                pd.to_datetime("{0}-{1}-01".format(*idxs)).daysinmonth
                for idxs in madt_ref.index]

            assert np.allclose(madt['MADT'], madt_ref['MADT'], rtol=1e-10,
                               equal_nan=True)
            assert np.allclose(madt['Days Available'],
                               madt_ref['Days Available'], rtol=1e-10,
                               equal_nan=True)
            assert np.allclose(madt['Days in Month'],
                               madt_ref['Days in Month'], rtol=1e-10,
                               equal_nan=True)

            # None of the sample data are for leap years.
            assert (madt['Days in Month'].sum() //
                    len(dca.index.levels[0])) == 365

    def test_get_aadt_py_from_madt(self, ptc_oneyear, ptc_multiyear):
        for ptc in (ptc_oneyear, ptc_multiyear):
            madt = self.dvc.get_madt(
                self.dvc.preprocess_daily_counts(ptc.data))
            aadt = self.dvc.get_aadt_py_from_madt(madt, ptc.perm_years)

            madt_py = madt.loc[ptc.perm_years, :].copy()
            madt_py['Weighted MADT'] = (madt_py['MADT'] *
                                        madt_py['Days in Month'])
            madtg = madt_py.groupby('Year')
            aadt_ref = (madtg['Weighted MADT'].sum() /
                        madtg['Days in Month'].sum())

            assert np.allclose(aadt['AADT'], aadt_ref, rtol=1e-10)

    def test_get_ratios_py(self, ptc_oneyear, ptc_multiyear):
        for ptc in (ptc_oneyear, ptc_multiyear):
            dca = self.dvc.preprocess_daily_counts(ptc.data)
            madt = self.dvc.get_madt(dca)
            aadt = self.dvc.get_aadt_py_from_madt(madt, ptc.perm_years)
            dom_ijd, d_ijd, n_avail_days = (
                self.dvc.get_ratios_py(dca, madt, aadt, ptc.perm_years))

            dc_dom = (dca.loc[ptc.perm_years]
                      .groupby(['Year', 'Month', 'Day of Week']))
            domadt = (dc_dom['Daily Count'].mean()
                      .unstack(level=-1, fill_value=np.nan))
            n_avail_days_ref = (dc_dom['Daily Count'].count()
                                .unstack(level=-1, fill_value=np.nan))

            assert np.allclose(n_avail_days, n_avail_days_ref,
                               rtol=1e-10, equal_nan=True)

            # Test if we can recover MADT from `domadt` and `dom_ijd`
            madt_pym = np.repeat(madt.loc[ptc.perm_years, 'MADT']
                                 .values[:, np.newaxis], 7, axis=1)
            madt_pym_est = (domadt * dom_ijd).values
            assert np.allclose(madt_pym_est[~np.isnan(madt_pym_est)],
                               madt_pym[~np.isnan(madt_pym_est)],
                               rtol=1e-10)

            # Test if we can recover AADT from `domadt` and `d_ijd`.
            aadt_pym = np.repeat(aadt['AADT']
                                 .values[:, np.newaxis], 7 * 12, axis=1)
            aadt_pym_est = (domadt * d_ijd).unstack(level=-1).values
            assert np.allclose(aadt_pym_est[~np.isnan(aadt_pym_est)],
                               aadt_pym[~np.isnan(aadt_pym_est)],
                               rtol=1e-10)


class TestDerivedValsStandard:

    def setup(self):
        self.dvc = dv.DerivedValsStandard()

    def test_get_derived_vals(self, sample_counts, cfgcm_test):
        ptc_oneyear = get_single_ptc(sample_counts, cfgcm_test, -890)
        ptc_multiyear = get_single_ptc(sample_counts, cfgcm_test, -104870)

        for ptc in (ptc_oneyear, ptc_multiyear):
            self.dvc.get_derived_vals(ptc)
            assert 'MADT' in ptc.adts.keys()
            assert 'AADT' in ptc.adts.keys()
            assert 'DoM_ijd' in ptc.ratios.keys()
            assert 'D_ijd' in ptc.ratios.keys()
            assert 'N_avail_days' in ptc.ratios.keys()

    def test_imputer(self, sample_counts, cfgcm_test):
        pass


class TestDerivedVals:

    def test_derivedvals(self):
        dvc = dv.DerivedVals('Standard')
        assert isinstance(dvc, dv.DerivedValsStandard)
        with pytest.raises(KeyError):
            dvc = dv.DerivedVals('Something')
