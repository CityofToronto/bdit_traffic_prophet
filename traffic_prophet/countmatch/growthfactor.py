"""Determine permanent count year-on-year growth factors."""

import numpy as np
import statsmodels.api as sm


GF_REGISTRY = {}
"""Dict for storing growth factor processor class definitions."""


class GFRegistrar(type):
    """Class registry processor, based on `baseband.vdif.header`.

    See https://github.com/mhvk/baseband.

    """

    _registry = GF_REGISTRY

    def __init__(cls, name, bases, dct):

        # Register GrowthFactorBase subclass if `_fit_type` not already taken.
        if name not in ('GrowthFactorBase', 'GrowthFactor'):
            if not hasattr(cls, "_fit_type"):
                raise ValueError("must define a `_fit_type`.")
            elif cls._fit_type in GFRegistrar._registry:
                raise ValueError("name {0} already registered in "
                                 "GF_REGISTRY".format(cls._fit_type))

            GFRegistrar._registry[cls._fit_type] = cls

        super().__init__(name, bases, dct)


class GrowthFactor:

    def __new__(cls, proctype, *args, **kwargs):
        # __init__ has to be called manually!
        self = super().__new__(GF_REGISTRY[proctype])
        self.__init__(*args, **kwargs)
        return self


class GrowthFactorBase(metaclass=GFRegistrar):
    """Base class for calculating growth factors."""

    @staticmethod
    def get_aadt(tc):
        """Get AADT (for permanent years)."""
        aadt = tc.adts['AADT'].reset_index()
        aadt['Year'] = aadt['Year'].astype(float)
        return aadt

    @staticmethod
    def get_wadt_py(tc):
        """Get WADT for all full weeks within permanent years."""
        cdata = tc.data.loc[tc.perm_years, :].copy()

        # Corrective measure, as dt.week returns the "week ordinal".  See
        # https://stackoverflow.com/a/55890652
        cdata['Week'] = cdata['Date'].dt.week
        cdata['Month'] = cdata['Date'].dt.month
        invalid_dates = (((cdata['Week'] == 1) & (cdata['Month'] == 12)) |
                         (cdata['Week'] == 53))
        cdata = cdata.loc[~invalid_dates, :]

        wadt = (cdata.groupby(['Year', 'Week'])['Daily Count']
                .agg(['mean', 'count'])).rename(columns={'mean': 'WADT'})

        # Only keep full weeks.
        wadt = wadt.loc[wadt['count'] == 7, ('WADT',)].reset_index()
        wadt['Time'] = wadt['Week'] + 52. * (wadt['Year'] - wadt['Year'].min())
        return wadt

    def fit_growth(self, tc):
        raise NotImplementedError


class GrowthFactorAADTExp(GrowthFactorBase):

    _fit_type = 'AADTExp'

    @staticmethod
    def exponential_rate_fit(year, aadt, ref_vals):
        r"""Calculate year-on-year exponential growth rate :math:`r` for ADT.

        Parameters
        ----------
        year : numpy.ndarray
            Array of year values.
        aadt : pandas.DataFrame
            Array of AADT values.
        ref_vals : dict containing 'year' and 'aadt' entries.
            Reference values for normalization.

        Returns
        -------
        r : statsmodels.regression.linear_model.RegressionResultsWrapper
            Fit results.

        Notes
        -----
        AADT for year :math:`y`, :math:`A_\mathrm{t}` is assumed to follow

        ..math::
            A_\mathrm{y} = A_{y_0} \exp{(r(y - y_0))}

        Where :math:`A_{y_0}` is some baseline AADT and :math:`r` the
        growthrate in units of :math:`1 / year`.

        See https://www.unescap.org/sites/default/files/Stats_Brief_Apr2015_Issue_07_Average-growth-rate.pdf
        and https://datahelpdesk.worldbank.org/knowledgebase/articles/906531-methodologies

        """
        # TO DO: consider relaxing the strict requirement that only r be
        # allowed to vary?

        # Dependent variable first.
        # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html
        model = sm.OLS(endog=(np.log(aadt / ref_vals['aadt'])),
                       exog=(year - ref_vals['year']))
        return model.fit()

    def fit_growth(self, tc):
        # Process year vs. AADT data.
        aadt = self.get_aadt(tc)

        # Perform exponential fit.
        fit_results = self.exponential_rate_fit(
            aadt['Year'].values,
            aadt['AADT'].values,
            {'year': aadt.at[0, 'Year'], 'aadt': aadt.at[0, 'AADT']})
        growth_factor = np.exp(fit_results.params[0])

        return {'fit_type': 'Exponential',
                'fit_results': fit_results,
                'growth_factor': growth_factor}


class GrowthFactorWADTLin(GrowthFactorBase):

    _fit_type = 'WADTLin'

    @staticmethod
    def linear_rate_fit(week, wadt):
        r"""OLS regression to estimate the linear growth rate in a single year.

        Parameters
        ----------
        week : numpy.ndarray
            Array of year values.
        aadt : pandas.DataFrame
            Array of AADT values.
        ref_vals : dict containing 'year' and 'aadt' entries.
            Reference values for normalization.

        Returns
        -------
        r : statsmodels.regression.linear_model.RegressionResultsWrapper
            Fit results.

        Notes
        -----
        ADT for week :math:`t`, :math:`A_\mathrm{t}` is assumed to follow

        ..math::
            A_\mathrm{t} = r(t - t_0) + A_{t_0}

        Where :math:`A_{t_0}` is ADT for a reference year and :math:`r` the
        growth rate in units of :math:`A / week`.

        """
        model = sm.OLS(endog=wadt, exog=sm.add_constant(week))
        return model.fit()

    def fit_growth(self, tc):
        # Process week vs. weekly averaged ADT.
        wadt_py = self.get_wadt_py(tc)
        fit_results = self.linear_rate_fit(wadt_py['Time'].values,
                                           wadt_py['WADT'].values)

        # Convert linear weekly fit to yearly exponential fit (iffy logic).
        # AADTs can only be calculated for perm_years, so consistent with
        # `get_wadt_py`.
        growth_factor = 1. + (fit_results.params[1] * 52. /
                              tc.adts['AADT']['AADT'].values[0])

        return {'fit_type': 'Linear',
                'fit_results': fit_results,
                'growth_factor': growth_factor}


class GrowthFactorComposite(GrowthFactorBase):
    # Not inheriting from GrowthFactorAADTExp and WADTLin because we need to
    # resolve two versions of `fit_growth`.

    _fit_type = 'Composite'

    def __init__(self):
        self.aadt_exp = GrowthFactorAADTExp()
        self.wadt_lin = GrowthFactorWADTLin()

    def fit_growth(self, tc):
        if tc.adts['AADT'].shape[0] > 1:
            return self.aadt_exp.fit_growth(tc)
        return self.wadt_lin.fit_growth(tc)
