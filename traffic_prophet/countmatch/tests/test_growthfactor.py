import pytest
import hypothesis as hyp
import numpy as np
import statsmodels.api as sm

from .. import permcount as pc
from .. import derivedvals as dv
from .. import growthfactor as gf


def get_single_ptc(counts, cfgcm, count_id):
    pcpp = pc.PermCountProcessor(None, None, cfg=cfgcm)
    perm_years = pcpp.partition_years(counts.counts[count_id])
    ptc = pc.PermCount.from_count_object(counts.counts[count_id],
                                         perm_years)
    dvs = dv.DerivedVals('Standard')
    dvs.get_derived_vals(ptc)
    return ptc


class TestGFRegistrarGrowthFactor:
    """Tests GFRegistrar and GrowthFactor."""

    def test_gfregistrar(self):

        # Test successful initialization of GrowthFactor subclass.
        class GrowthFactorCompositeTest(gf.GrowthFactorComposite):
            _fit_type = 'Testing'

        assert gf.GF_REGISTRY['Testing'] is GrowthFactorCompositeTest
        gf_instance = gf.GrowthFactor('Testing')
        assert gf_instance._fit_type == 'Testing'

        # Pop the dummy class, in case we test twice.
        gf.GFRegistrar._registry.pop('Testing', None)

        # Test repeated `_dv_type` error handling.
        with pytest.raises(ValueError) as excinfo:
            class GrowthFactorCompositeBad1(gf.GrowthFactorComposite):
                pass
        assert "already registered in" in str(excinfo.value)

        # Test missing `_dv_type` error handling.
        with pytest.raises(ValueError) as excinfo:
            class GrowthFactorCompositeBad2(gf.GrowthFactorBase):
                pass
        assert "must define a" in str(excinfo.value)


class TestGrowthFactorBase:
    """Test growth factor base class."""

    @pytest.fixture()
    def ptc_oneyear(self, sample_counts, cfgcm_test):
        return get_single_ptc(sample_counts, cfgcm_test, -890)

    @pytest.fixture()
    def ptc_multiyear(self, sample_counts, cfgcm_test):
        return get_single_ptc(sample_counts, cfgcm_test, -104870)

    def setup(self):
        self.gfb = gf.GrowthFactorBase()

    def test_get_aadt(self, ptc_oneyear, ptc_multiyear):
        for ptc in (ptc_oneyear, ptc_multiyear):
            aadt = self.gfb.get_aadt(ptc)
            assert list(np.sort(aadt.columns)) == ['AADT', 'Year']
            assert aadt['Year'].dtype == np.dtype('float64')
            assert np.array_equal(ptc.adts['AADT'].index.values,
                                  aadt['Year'].values)
            assert np.array_equal(ptc.adts['AADT']['AADT'].values,
                                  aadt['AADT'].values)

    def test_get_wadt_py(self, ptc_oneyear, ptc_multiyear):
        # Confirm WADT values for individual weeks.
        wadt_oy = self.gfb.get_wadt_py(ptc_oneyear)
        wadt_jun14 = (ptc_oneyear.data
                      .loc[(2010, 165):(2010, 171), 'Daily Count'].mean())
        wadt_nov29 = (ptc_oneyear.data
                      .loc[(2010, 333):(2010, 339), 'Daily Count'].mean())
        assert np.isclose(
            wadt_oy.loc[wadt_oy['Week'] == 24, 'WADT'].values[0], wadt_jun14)
        assert np.isclose(
            wadt_oy.loc[wadt_oy['Week'] == 48, 'WADT'].values[0], wadt_nov29)

        wadt_my = self.gfb.get_wadt_py(ptc_multiyear)

        wadt_apr26_2010 = (ptc_multiyear.data
                           .loc[(2010, 116):(2010, 122), :])
        wadt_my_apr26_2010 = wadt_my.loc[
            (wadt_my['Year'] == 2010) & (wadt_my['Week'] == 17), :]
        assert np.allclose(
            wadt_my_apr26_2010[['WADT', 'Time']].values.ravel(),
            np.array([wadt_apr26_2010['Daily Count'].mean(), 17.]))

        wadt_oct15_2012 = (ptc_multiyear.data
                           .loc[(2012, 289):(2012, 295), :])
        wadt_my_oct15_2012 = wadt_my.loc[
            (wadt_my['Year'] == 2012) & (wadt_my['Week'] == 42), :]
        assert np.allclose(
            wadt_my_oct15_2012[['WADT', 'Time']].values.ravel(),
            np.array([wadt_oct15_2012['Daily Count'].mean(), 146.]))


class TestGrowthFactorAADTExp:

    def setup(self):
        self.gfc = gf.GrowthFactorAADTExp()

    @hyp.given(slp=hyp.strategies.floats(min_value=-2., max_value=2.))
    @hyp.settings(max_examples=30)
    def test_exponential_rate_fit(self, slp):
        # Create a generic exponential curve.
        x = np.linspace(1.5, 2.7, 100)
        y = np.exp(slp * x)
        result = self.gfc.exponential_rate_fit(
            x, y, {"year": x[0], "aadt": y[0]})
        assert np.abs(result.params[0] - slp) < 0.01

    def test_fit_growth(self, sample_counts, cfgcm_test):
        ptc_multiyear = get_single_ptc(sample_counts, cfgcm_test, -104870)

        aadt = self.gfc.get_aadt(ptc_multiyear)
        fit_ref = sm.OLS(np.log(aadt['AADT'].values / aadt['AADT'].values[0]),
                         aadt['Year'].values - aadt['Year'].values[0]).fit()

        self.gfc.fit_growth(ptc_multiyear)

        assert np.isclose(ptc_multiyear._growth_fit["growth_factor"],
                          np.exp(fit_ref.params[0]))
        assert (ptc_multiyear.growth_factor == 
                ptc_multiyear._growth_fit["growth_factor"])


class TestGrowthFactorWADTLin:

    def setup(self):
        self.gfc = gf.GrowthFactorWADTLin()

    @hyp.given(slp=hyp.strategies.floats(min_value=-2., max_value=2.),
               y0=hyp.strategies.floats(min_value=-2., max_value=2.))
    @hyp.settings(max_examples=30)
    def test_linear_rate_fit(self, slp, y0):
        # Create a generic line.
        x = np.linspace(0.5, 3.7, 100)
        y = slp * x + y0
        result = self.gfc.linear_rate_fit(x, y)
        assert np.abs(result.params[1] - slp) < 0.01

    def test_fit_growth(self, sample_counts, cfgcm_test):
        ptc_oneyear = get_single_ptc(sample_counts, cfgcm_test, -890)
        ptc_multiyear = get_single_ptc(sample_counts, cfgcm_test, -104870)

        # The multi-year fit is REALLY sketchy, since we end up normalizing
        # by the first year's AADT.
        for ptc in (ptc_oneyear, ptc_multiyear):
            wadt = self.gfc.get_wadt_py(ptc)
            fit_ref = sm.OLS(wadt['WADT'].values,
                             sm.add_constant(wadt['Time'].values)).fit()

            self.gfc.fit_growth(ptc)

            assert np.isclose(ptc._growth_fit["growth_factor"],
                              1. + (fit_ref.params[1] * 52. /
                                    ptc.adts['AADT']['AADT'].values[0]))
            assert (ptc.growth_factor ==
                    ptc._growth_fit["growth_factor"])


class TestGrowthFactorComposite:

    def test_growthfactorcomposite(self, sample_counts, cfgcm_test):
        ptc_oneyear = get_single_ptc(sample_counts, cfgcm_test, -890)
        ptc_multiyear = get_single_ptc(sample_counts, cfgcm_test, -104870)
        gfc = gf.GrowthFactorComposite()

        gfc.fit_growth(ptc_oneyear)
        gfc.fit_growth(ptc_multiyear)
        assert ptc_oneyear._growth_fit['fit_type'] == 'Linear'
        assert ptc_multiyear._growth_fit['fit_type'] == 'Exponential'


class TestGrowthFactor:

    def test_growthfactor(self):
        gfc = gf.GrowthFactor('Composite')
        assert isinstance(gfc, gf.GrowthFactorComposite)
        gfc = gf.GrowthFactor('AADTExp')
        assert isinstance(gfc, gf.GrowthFactorAADTExp)
        with pytest.raises(KeyError):
            gfc = gf.GrowthFactor('Base')
