"""Determine permanent count year-on-year growth factors."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

from . import reader


def exponential_growth_factor(domadt):
    """Calculate year-on-year exponential growth rate :math:`r`.

    ADT for (integer) year :math:`y`, :math:`A_\mathrm{y}` is assumed to follow

    ..math::
        A_\mathrm{y} = A_{y_0} \exp{(r(y - y_0))}

    Where :math:`A_{y_0}` is some baseline ADT and :math:`r` the year-on-year
    growth rate.

    Parameters
    ----------
    domadt : pandas.DataFrame
        DataFrame of day-of-month ADT.
    """
    # See https://www.unescap.org/sites/default/files/Stats_Brief_Apr2015_Issue_07_Average-growth-rate.pdf
    # and https://datahelpdesk.worldbank.org/knowledgebase/articles/906531-methodologies
    pass

def linear_growth_factor(domadt):
    """OLS regression to estimate the linear growth rate.

    ADT for (integer) year :math:`y`, :math:`A_\mathrm{y}` is assumed to follow

    ..math::
        A_\mathrm{y} = r(y - y_0) + A_{y_0}

    Where :math:`A_{y_0}` is some baseline ADT and :math:`r` the year-on-year
    growth rate.

    In practice, used when there is only one year of data.

    Parameters
    ----------
    domadt : pandas.DataFrame
        DataFrame of day-of-month ADT.

    """
    y = uberdrivers_all['num_distinct_drivers'].values.astype('float')
    X = uberdrivers_all['number_of_trips'].values.astype('float')
    model = sm.ols(formula="y ~ X", data={'X': X, 'y': y})
    results_lin = model.fit()

    results_lin.summary()



class PermCount(reader.Count):
    """Class to hold permanent count data and calculate growth factors."""

    def __init__(self, centreline_id, direction, data):
        super().__init__(centreline_id, direction, data,
                         is_permanent=True)
        self.growth = None

    @classmethod
    def from_ptc_count_object(cls, ptc):
        # Data will be passed by reference.
        return cls(ptc.centreline_id, ptc.direction, ptc.data)

    def get_growth_factor(self):
        domadt = self.data['DoMADT']
        if len(domadt.index.levels[0]) > 1:
            self.growth = exponential_growth_factor(domadt)
        else:
            self.growth = linear_growth_factor(domadt)


def get_growth_factors(rdr):
    for key in rdr.ptcs.keys():
        rdr.ptcs[key] = PermCount.from_ptc_count_object(rdr.ptcs[key])
        rdr.ptcs[key].get_growth_factor()