import pytest
import numpy as np
import pandas as pd

from .. import permcount as pc
from .. import derivedvals as dv


def get_single_ptc(counts, cfgcm, count_id):
    pcpp = pc.PermCountProcessor(None, None, cfg=cfgcm)
    perm_years = pcpp.partition_years(counts.counts[count_id])
    ptc = pc.PermCount.from_count_object(counts.counts[count_id], perm_years)
    return ptc


class TestDVRegistrarDerivedVals:
    """Tests DVRegistrar and DerivedVals."""

    def test_dvregistrar(self):

        # Test successful initialization of DerivedVals subclass.
        class DerivedValsStandardTest(dv.DerivedValsStandard):
            _dv_type = 'Testing'

        assert dv.DV_REGISTRY['Testing'] is DerivedValsStandardTest
        dv_instance = dv.DerivedVals('Testing')
        assert dv_instance._dv_type == 'Testing'

        # Pop the dummy class, in case we test twice.
        dv.DVRegistrar._registry.pop('Testing', None)

        # Test repeated `_dv_type` error handling.
        with pytest.raises(ValueError) as excinfo:
            class DerivedValsStandardBad1(dv.DerivedValsStandard):
                pass
        assert "already registered in" in str(excinfo.value)

        # Test missing `_dv_type` error handling.
        with pytest.raises(ValueError) as excinfo:
            class DerivedValsStandardBad2(dv.DerivedValsBase):
                pass
        assert "must define a" in str(excinfo.value)


class TestDerivedValsBase:

    @pytest.fixture(params=[-890, -104870])
    def ptc_sample(self, sample_counts, cfgcm_test, request):
        return get_single_ptc(sample_counts, cfgcm_test, request.param)

    def setup(self):
        self.dvc = dv.DerivedValsBase()

    def test_preprocess_daily_counts(self, ptc_sample):
        dca = self.dvc.preprocess_daily_counts(ptc_sample.data)
        assert np.array_equal(dca['Month'], ptc_sample.data['Date'].dt.month)
        assert np.array_equal(dca['Day of Week'],
                              ptc_sample.data['Date'].dt.dayofweek)

    def test_get_madt(self, ptc_sample):
        dca = self.dvc.preprocess_daily_counts(ptc_sample.data)
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

        tols = {'rtol': 1e-10, 'equal_nan': True}
        assert np.allclose(madt['MADT'], madt_ref['MADT'], **tols)
        assert np.allclose(madt['Days Available'],
                           madt_ref['Days Available'], **tols)
        assert np.allclose(madt['Days in Month'],
                           madt_ref['Days in Month'], **tols)

        # None of the sample data are for leap years.
        assert (madt['Days in Month'].sum() //
                len(dca.index.levels[0])) == 365
        assert (madt['Days in Month'].sum() % len(dca.index.levels[0])) == (
            1 if 2012 in dca.index.levels[0] else 0)

    def test_get_aadt_py_from_madt(self, ptc_sample):
        madt = self.dvc.get_madt(
            self.dvc.preprocess_daily_counts(ptc_sample.data))
        aadt = self.dvc.get_aadt_py_from_madt(madt, ptc_sample.perm_years)

        madt_py = madt.loc[ptc_sample.perm_years, :].copy()
        madt_py['Weighted MADT'] = (madt_py['MADT'] *
                                    madt_py['Days in Month'])
        madtg = madt_py.groupby('Year')
        aadt_ref = (madtg['Weighted MADT'].sum() /
                    madtg['Days in Month'].sum())

        assert np.allclose(aadt['AADT'], aadt_ref, rtol=1e-10)

    def test_get_ratios_py(self, ptc_sample):
        dca = self.dvc.preprocess_daily_counts(ptc_sample.data)
        madt = self.dvc.get_madt(dca)
        aadt = self.dvc.get_aadt_py_from_madt(madt, ptc_sample.perm_years)
        dom_ijd, d_ijd, n_avail_days = (
            self.dvc.get_ratios_py(dca, madt, aadt, ptc_sample.perm_years))

        dc_dom = (dca.loc[ptc_sample.perm_years]
                  .groupby(['Year', 'Month', 'Day of Week']))
        domadt = (dc_dom['Daily Count'].mean()
                  .unstack(level=-1, fill_value=np.nan))
        n_avail_days_ref = (dc_dom['Daily Count'].count()
                            .unstack(level=-1, fill_value=np.nan))

        assert np.allclose(n_avail_days, n_avail_days_ref,
                           rtol=1e-10, equal_nan=True)

        # Test if we can recover MADT from `domadt` and `dom_ijd`.
        madt_pym = np.repeat(madt.loc[ptc_sample.perm_years, 'MADT']
                             .values[:, np.newaxis], 7, axis=1)
        madt_pym_est = (domadt * dom_ijd).values
        # madt_pym naturally has no NaNs, while madt_pym_est does, so only
        # compare non-NaN values.
        assert np.allclose(madt_pym_est[~np.isnan(madt_pym_est)],
                           madt_pym[~np.isnan(madt_pym_est)],
                           rtol=1e-10)

        # Test if we can recover AADT from `domadt` and `d_ijd`.
        aadt_pym = np.repeat(aadt['AADT']
                             .values[:, np.newaxis], 7 * 12, axis=1)
        aadt_pym_est = (domadt * d_ijd).unstack(level=-1).values
        # aadt_pym naturally has no NaNs, while aadt_pym_est does, so only
        # compare non-NaN values.
        assert np.allclose(aadt_pym_est[~np.isnan(aadt_pym_est)],
                           aadt_pym[~np.isnan(aadt_pym_est)],
                           rtol=1e-10)


class TestDerivedValsStandard:

    def setup(self):
        self.dvc = dv.DerivedValsStandard()

    @pytest.mark.parametrize('count_id', [-890, -104870])
    def test_get_derived_vals(self, sample_counts, cfgcm_test, count_id):
        ptc = get_single_ptc(sample_counts, cfgcm_test, count_id)

        self.dvc.get_derived_vals(ptc)
        assert sorted(list(ptc.adts.keys())) == ['AADT', 'MADT']
        assert sorted(list(ptc.ratios.keys())) == [
            'D_ijd', 'DoM_ijd', 'N_avail_days']

    def test_imputer(self, sample_counts, cfgcm_test):
        pass


class TestDerivedVals:

    def test_derivedvals(self):
        dvc = dv.DerivedVals('Standard')
        assert isinstance(dvc, dv.DerivedValsStandard)
        with pytest.raises(KeyError):
            dvc = dv.DerivedVals('Something')
