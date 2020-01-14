"""Determine derived values (DoM Factors, etc.) from permanent counts."""

import numpy as np
import pandas as pd


class DerivedValBase:
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
        dom_ijd = madt['MADT'].values[:, np.newaxis] / domadt
        # Determine day-to-year conversion factor D_ijd.
        d_ijd = (aadt['AADT'].values[:, np.newaxis] /
                 domadt.loc[perm_years, :].unstack(level=-1)).stack()

        return dom_ijd, d_ijd, n_avail_days

    def get_derived_vals(self, ptc):
        raise NotImplementedError


class DerivedValSimple(DerivedValBase):

    @staticmethod
    def get_derived_vals(ptc):
        pass