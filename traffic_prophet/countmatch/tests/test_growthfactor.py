import hypothesis as hyp
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .. import reader
from .. import growthfactor as gf

from ...data import SAMPLE_ZIP


class TestGrowthFactor:
    """Test growth factor calculation."""

    @hyp.given(slp=hyp.strategies.floats(min_value=-2., max_value=2.))
    @hyp.settings(max_examples=30)
    def test_exponential_factor_fit(self, slp):
        # Create a generic exponential curve.
        x = np.linspace(1.5, 2.7, 100)
        y = np.exp(slp * x)
        result = gf.exponential_factor_fit(x, y,
                                           {"year": x[0], "aadt": y[0]})
        assert np.abs(result.params[0] - slp) < 0.01

    @hyp.given(slp=hyp.strategies.floats(min_value=-2., max_value=2.),
               y0=hyp.strategies.floats(min_value=-2., max_value=2.))
    @hyp.settings(max_examples=30)
    def test_linear_factor_fit(self, slp, y0):
        # Create a generic line.
        x = np.linspace(0.5, 3.7, 100)
        y = slp * x + y0
        result = gf.linear_factor_fit(x, y)
        assert np.abs(result.params[1] - slp) < 0.01

    def test_perm_count(self):

        # Read in data (not doing this at setup since we'll be altering rdr in
        # another test.
        rdr = reader.Reader(SAMPLE_ZIP)
        rdr.read()

        # PTC -104870 has multiple years of data.
        ptc_raw = rdr.ptcs[-104870]
        ptc_perm = gf.PermCount.from_ptc_count_object(ptc_raw)
        assert ptc_perm.centreline_id == ptc_raw.centreline_id
        assert ptc_perm.direction == ptc_raw.direction
        # Python is beautiful - this does a deep compare!
        # https://stackoverflow.com/questions/1911273/is-there-a-better-way-to-compare-dictionary-values/5635309#5635309
        assert ptc_perm.data == ptc_raw.data

        # Check AADT retrieval.
        aadt = ptc_perm.get_aadt()
        assert np.array_equal(ptc_perm.data['AADT'].index.values,
                              aadt['Year'].values)
        assert np.array_equal(ptc_perm.data['AADT']['AADT'].values,
                              aadt['AADT'].values)

        # Check fitter.
        fit = sm.OLS(np.log(aadt['AADT'].values / aadt['AADT'].values[0]),
                     aadt['Year'].values - aadt['Year'].values[0]).fit()
        ptc_perm.fit_growth()
        assert ptc_perm._fit_type == 'Exponential'
        assert np.isclose(ptc_perm.growth_factor, fit.params[0])

        # PTC -890 only has one year.
        ptc_raw = rdr.ptcs[-890]
        ptc_perm = gf.PermCount.from_ptc_count_object(ptc_raw)

        # Check WADT retrieval.
        wadt = ptc_perm.get_wadt()
        cdata_week = pd.DataFrame(
            {'Week': ptc_raw.data['Daily Count']['Date'].dt.week})
        avail_week = (cdata_week.loc[cdata_week['Week'] < 53, :]
                      .groupby('Week')['Week'].count())
        # Check that all weeks with 7 days are in wadt.
        assert np.array_equal(avail_week[avail_week == 7].index.values,
                              wadt['Week'].unique().astype(int))
        # Check WADT values.
        wadt_jun14 = (ptc_raw.data['Daily Count']
                      .loc[(2010, 165):(2010, 171), 'Daily Count'].mean())
        wadt_nov29 = (ptc_raw.data['Daily Count']
                      .loc[(2010, 333):(2010, 339), 'Daily Count'].mean())
        assert np.isclose(
            (wadt.loc[wadt['Start of Week'] == '2010-06-14', 'WADT']
             .values[0]), wadt_jun14)
        assert np.isclose(
            (wadt.loc[wadt['Start of Week'] == '2010-11-29', 'WADT']
             .values[0]), wadt_nov29)

        # Check fit values.
        fit = sm.OLS(wadt['WADT'].values,
                     sm.add_constant(wadt['Week'].values)).fit()
        ptc_perm.fit_growth()
        assert ptc_perm._fit_type == 'Linear'
        assert np.isclose(ptc_perm.growth_factor,
                          fit.params[1] * 52. / ptc_raw.data['AADT'].iat[0, 0])

    def test_get_growth_factors(self):
        # Test that using gf.get_growth_factors actually cycles through all
        # permanent stations to calculate growth factors.
        rdr = reader.Reader(SAMPLE_ZIP)
        rdr.read()

        gf.get_growth_factors(rdr)
        for item in rdr.ptcs.values():
            assert isinstance(item, gf.PermCount)
            assert item.growth_factor is not None
