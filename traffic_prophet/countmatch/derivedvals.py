"""Determine derived values (DoM Factors, etc.) from permanent counts."""

import numpy as np
import pandas as pd


DV_REGISTRY = {}
"""Dict for storing derived value processor class definitions."""


class DVRegistrar(type):
    """Class registry processor, based on `baseband.vdif.header`.

    See https://github.com/mhvk/baseband.

    """

    _registry = DV_REGISTRY

    def __init__(cls, name, bases, dct):

        # Register GrowthFactorBase subclass if `_dv_type` not already taken.
        if name not in ('DerivedValsBase', 'DerivedVals'):
            if not hasattr(cls, "_dv_type"):
                raise ValueError("must define a `_dv_type`.")
            elif cls._dv_type in DVRegistrar._registry:
                raise ValueError("name {0} already registered in "
                                 "DV_REGISTRY".format(cls._dv_type))

            DVRegistrar._registry[cls._dv_type] = cls

        super().__init__(name, bases, dct)


class DerivedVals:

    def __new__(cls, dvtype, *args, **kwargs):
        # __init__ has to be called manually!
        # https://docs.python.org/3/reference/datamodel.html#object.__new__
        # https://stackoverflow.com/questions/20221858/python-new-method-returning-something-other-than-class-instance
        self = super().__new__(DV_REGISTRY[dvtype])
        self.__init__(*args, **kwargs)
        return self


class DerivedValsBase(metaclass=DVRegistrar):
    """Base class for getting derived values for permanent counts.

    Notes
    -----
    Averaging methods are more conservative than `STTC_estimate3.m` - only
    days with complete data are included in the MADT and DoMADT estimates.

    """

    _months_of_year = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    @staticmethod
    def preprocess_daily_counts(dc):
        # Ensure dc remains unchanged by the method.
        dca = dc.copy()
        dca['Month'] = dca['Date'].dt.month
        dca['Day of Week'] = dca['Date'].dt.dayofweek
        return dca

    def get_madt(self, dca):
        dc_m = dca.groupby(['Year', 'Month'])
        madt = pd.DataFrame({
            'MADT': dc_m['Daily Count'].mean(),
            'Days Available': dc_m['Daily Count'].count()},
            index=pd.MultiIndex.from_product(
                [dca.index.levels[0], np.arange(1, 13, dtype=int)],
                names=['Year', 'Month'])
        )

        # Loop to record number of days in month.
        days_in_month = []
        for year in dca.index.levels[0]:
            cdays = self._months_of_year.copy()
            cdays[1] = pd.to_datetime('{0:d}-02-01'.format(year)).daysinmonth
            days_in_month += cdays
        madt['Days in Month'] = days_in_month

        return madt

    @staticmethod
    def get_aadt_from_madt(madt, perm_years):
        # Weighted average for AADT.
        madt_py = madt.loc[perm_years, :]
        monthly_total_traffic = madt_py['MADT'] * madt_py['Days in Month']
        return pd.DataFrame(
            {'AADT': (monthly_total_traffic.groupby('Year').sum() /
                      madt_py.groupby('Year')['Days in Month'].sum())})

    @staticmethod
    def get_ratios(dca, madt, aadt, perm_years):
        dc_dom = dca.groupby(['Year', 'Month', 'Day of Week'])
        ymd_index = pd.MultiIndex.from_product(
            [dca.index.levels[0], np.arange(1, 13, dtype=int)],
            names=['Year', 'Month'])

        domadt = pd.DataFrame(
            dc_dom['Daily Count'].mean().unstack(level=-1, fill_value=np.nan),
            index=ymd_index)
        n_avail_days = pd.DataFrame(
            dc_dom['Daily Count'].count().unstack(level=-1, fill_value=np.nan),
            index=ymd_index)

        # Determine day-to-month conversion factor DoM_ijd.  (Uses a numpy
        # broadcasting trick.)
        dom_ijd = (madt['MADT'].loc[perm_years, :].values[:, np.newaxis] /
                   domadt.loc[perm_years, :])
        # Determine day-to-year conversion factor D_ijd.
        d_ijd = (aadt['AADT'].values[:, np.newaxis] /
                 domadt.loc[perm_years, :].unstack(level=-1)).stack()

        return domadt, dom_ijd, d_ijd, n_avail_days

    def get_derived_vals(self, ptc):
        raise NotImplementedError


class DerivedValsStandard(DerivedValsBase):

    _dv_type = 'Standard'

    def __init__(self, impute_ratios=False):
        self._impute_ratios = impute_ratios

    def get_derived_vals(self, ptc):
        dca = self.preprocess_daily_counts(ptc.data['Daily Count'])
        madt = self.get_madt(dca)
        aadt = self.get_aadt_from_madt(madt, ptc.perm_years)
        domadt, dom_ijd, d_ijd, n_avail_days = self.get_ratios(
            dca, madt, aadt, ptc.perm_years)

        ptc.data['MADT'] = madt
        ptc.data['AADT'] = aadt
        ptc.ratios = {'DoM_ijd': dom_ijd, 'D_ijd': d_ijd,
                      'N_avail_days': n_avail_days}
