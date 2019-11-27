"""Determine permanent count year-on-year growth factors."""

import numpy as np
import statsmodels.api as sm

from . import reader


def exponential_rate_fit(year, aadt, ref_vals):
    """Calculate year-on-year exponential growth rate :math:`r` for ADT.

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

    Where :math:`A_{y_0}` is some baseline AADT and :math:`r` the growth rate
    in units of :math:`1 / year`.

    See https://www.unescap.org/sites/default/files/Stats_Brief_Apr2015_Issue_07_Average-growth-rate.pdf
    and https://datahelpdesk.worldbank.org/knowledgebase/articles/906531-methodologies
    """
    # TO DO: consider relaxing the strict requirement that only r be allowed to
    # vary?

    # Dependent variable first.
    # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html
    model = sm.OLS(endog=(np.log(aadt / ref_vals['aadt'])),
                   exog=(year - ref_vals['year']))
    return model.fit()


def linear_rate_fit(week, wadt):
    """OLS regression to estimate the linear growth rate within a single year.

    ADT for week :math:`t`, :math:`A_\mathrm{t}` is assumed to follow

    ..math::
        A_\mathrm{t} = r(t - t_0) + A_{t_0}

    Where :math:`A_{t_0}` is ADT for a reference year and :math:`r` the
    growth rate in units of :math:`A / week`.

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
    """
    model = sm.OLS(endog=wadt, exog=sm.add_constant(week))
    return model.fit()


class PermCount(reader.Count):
    """Class to hold permanent count data and calculate growth factors."""

    def __init__(self, count_id, centreline_id, direction, data):
        super().__init__(count_id, centreline_id, direction, data,
                         is_permanent=True)
        self.growth_factor = None
        self.base_year = None
        self._fit = None
        self._fit_type = None

    @classmethod
    def from_ptc_count_object(cls, ptc):
        # Data will be passed by reference.
        return cls(ptc.count_id, ptc.centreline_id, ptc.direction, ptc.data)

    def get_aadt(self):
        aadt = self.data['AADT'].reset_index()
        aadt['Year'] = aadt['Year'].astype(float)
        return aadt

    def get_wadt(self):
        cdata = self.data['Daily Count'].reset_index()
        # Overcomplicated groupby using the start of the week, as dt.week
        # returns the "week ordinal".  See https://stackoverflow.com/a/55890652
        cdata['Start of Week'] = (
            cdata['Date'] -
            cdata['Date'].dt.dayofweek * np.timedelta64(1, 'D'))
        wadt = (cdata.groupby('Start of Week')['Daily Count']
                .agg(['mean', 'count']))
        wadt = wadt.loc[wadt['count'] == 7, ('mean',)]
        wadt.reset_index(inplace=True)
        wadt.columns = ('Start of Week', 'WADT')
        wadt['Week'] = wadt['Start of Week'].dt.week.astype(float)
        return wadt

    def fit_growth(self):
        if len(self.data['DoMADT'].index.levels[0]) > 1:
            self._fit_type = 'Exponential'

            # Process year vs. AADT data.
            aadt = self.get_aadt()

            # Perform exponential fit.
            self._fit = exponential_rate_fit(
                aadt['Year'].values,
                aadt['AADT'].values,
                {'year': aadt.at[0, 'Year'], 'aadt': aadt.at[0, 'AADT']})

            # Populate growth factor.
            self.growth_factor = np.exp(self._fit.params[0])
            self.base_year = int(aadt.at[0, 'Year'])
        else:
            self._fit_type = 'Linear'

            # Process week vs. weekly averaged ADT.
            wadt = self.get_wadt()
            self._fit = linear_rate_fit(wadt['Week'].values,
                                        wadt['WADT'].values)

            # Convert linear weekly fit to yearly exponential fit (iffy logic).
            aadt_info = self.data['AADT'].reset_index()
            self.growth_factor = 1. + (self._fit.params[1] * 52. /
                                       aadt_info['AADT'].values[0])
            self.base_year = aadt_info['Year'].values[0]


def get_growth_factors(rdr):
    for key in rdr.ptcs.keys():
        rdr.ptcs[key] = PermCount.from_ptc_count_object(rdr.ptcs[key])
        rdr.ptcs[key].fit_growth()
