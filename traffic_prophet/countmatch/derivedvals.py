"""Determine derived values (DoM Factors, etc.) from permanent counts."""

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn import impute as skimp


DV_REGISTRY = {}
"""Dict for storing derived value processor class definitions."""


class DVRegistrar(type):
    """Class registry processor, based on `baseband.vdif.header`.

    See https://github.com/mhvk/baseband.

    """

    _registry = DV_REGISTRY

    def __init__(cls, name, bases, dct):

        # Register DerivedVal subclass if `_dv_type` not already taken.
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
        """Preprocess daily counts by adding month and day of week.

        Parameters
        ---------------
        dc : pandas.DataFrame
            Daily counts.

        Returns
        -------
        dca : pandas.DataFrame
            Copy of `dc` with 'Month' and 'Day of Week' columns.

        """
        # Ensure dc remains unchanged by the method.
        dca = dc.copy()
        dca['Month'] = dca['Date'].dt.month
        dca['Day of Week'] = dca['Date'].dt.dayofweek
        return dca

    def get_madt(self, dca):
        """Get mean average daily traffic (MADT) from processed daily counts.

        Parameters
        ---------------
        dca : pandas.DataFrame
            Daily counts, with 'Month' and 'Day of Week' columns from
            `preprocess_daily_counts`.

        Returns
        -------
        madt : pandas.DataFrame
            MADT table, with 'Days Available' and 'Days in Month' columns.

        """
        dc_m = dca.groupby(['Year', 'Month'])
        madt = pd.DataFrame({
            'MADT': dc_m['Daily Count'].mean(),
            'Days Available': dc_m['Daily Count'].count()},
            index=pd.MultiIndex.from_product(
                [dca.index.levels[0], np.arange(1, 13, dtype=int)],
                names=['Year', 'Month'])
        )

        # Loop to record the number of days in each month. February requires
        # recalculating in case of leap years.
        days_in_month = []
        for year in madt.index.levels[0]:
            cdays = self._months_of_year.copy()
            cdays[1] = pd.to_datetime('{0:d}-02-01'.format(year)).daysinmonth
            days_in_month += cdays
        madt['Days in Month'] = days_in_month

        return madt

    @staticmethod
    def get_aadt_py_from_madt(madt, perm_years):
        """Annual average daily traffic (AADT) from an MADT weighted average.

        Parameters
        ---------------
        madt : pandas.DataFrame
            MADT, with 'Days in Month' column as from `get_madt`.
        perm_years : list
            List of permanent count years for location; obtained from
             PermCount.perm_years.

        Returns
        -------
        aadt : pandas.DataFrame
            AADT table.

        """
        madt_py = madt.loc[perm_years, :]
        monthly_total_traffic = madt_py['MADT'] * madt_py['Days in Month']
        return pd.DataFrame(
            {'AADT': (monthly_total_traffic.groupby('Year').sum() /
                      madt_py.groupby('Year')['Days in Month'].sum())})

    @staticmethod
    def get_ratios_py(dca, madt, aadt_py, perm_years):
        """Ratios between MADT and AADT and day-of-month average daily traffic.

        Parameters
        ---------------
        dca : pandas.DataFrame
            Daily counts, with 'Month' and 'Day of Week' columns from
            `preprocess_daily_counts`.
        madt : pandas.DataFrame
            MADT, with 'Days in Month' column as from `get_madt`.
        aadt_py : pandas.DataFrame
            AADT for permanent years, as from `get_aadt_py_from_madt`.
        perm_years : list
            List of permanent count years for location; obtained from
            PermCount.perm_years.

        Returns
        -------
        dom_ijd : pandas.DataFrame
            Ratio between MADT and day-of-month ADT.
        d_ijd : pandas.DataFrame
            Ratio between AADT and day-of-month ADT.
        n_avail_days : pandas.DataFrame
            Number of days used to calculate day-of-month ADT.

        """
        dc_dom = dca.loc[perm_years].groupby(['Year', 'Month', 'Day of Week'])
        # Multi-index levels retain all values from dca even after using loc,
        # so can only use `perm_years`.
        ymd_index = pd.MultiIndex.from_product(
            [perm_years, np.arange(1, 13, dtype=int)], names=['Year', 'Month'])

        domadt = pd.DataFrame(
            dc_dom['Daily Count'].mean().unstack(level=-1, fill_value=np.nan),
            index=ymd_index)
        n_avail_days = pd.DataFrame(
            dc_dom['Daily Count'].count().unstack(level=-1, fill_value=np.nan),
            index=ymd_index)

        # Determine day-to-month conversion factor DoM_ijd.  (Uses a numpy
        # broadcasting trick.)
        dom_ijd = (madt['MADT'].loc[perm_years, :].values[:, np.newaxis] /
                   domadt)
        # Determine day-to-year conversion factor D_ijd.  (Uses broadcasting
        # and pivoting pandas columns.)
        d_ijd = (aadt_py['AADT'].values[:, np.newaxis] /
                 domadt.unstack(level=-1)).stack()

        return dom_ijd, d_ijd, n_avail_days

    def get_derived_vals(self, ptc):
        raise NotImplementedError


class DerivedValsStandard(DerivedValsBase):

    _dv_type = 'Standard'

    def __init__(self, impute_ratios=False, **kwargs):
        self._impute_ratios = impute_ratios
        self._imputer_args = kwargs

    def get_derived_vals(self, ptc):
        """Get derived values, including ADTs and ratios between them.

        Depending on settings, will also impute missing values.

        Parameters
        ----------
        ptc : permcount.PermCount
            Permanent count instance.

        """
        dca = self.preprocess_daily_counts(ptc.data)
        madt = self.get_madt(dca)
        aadt = self.get_aadt_py_from_madt(madt, ptc.perm_years)
        dom_ijd, d_ijd, n_avail_days = self.get_ratios_py(
            dca, madt, aadt, ptc.perm_years)

        ptc.adts = {'MADT': madt, 'AADT': aadt}
        ptc.ratios = {'DoM_ijd': dom_ijd, 'D_ijd': d_ijd,
                      'N_avail_days': n_avail_days}

        if self._impute_ratios:
            self.impute_ratios(ptc)

    @staticmethod
    def fill_nans(df, imp):
        """Fill NaN values in an array with imputed ones.

        Parameters
        ----------
        df : pandas.DataFrame
            Original data, with NaNs.
        imp : numpy.ndarray
            Data array with imputed values.

        """
        for i, j in zip(*np.where(df.isnull())):
            df.iloc[i, j] = imp[i, j]

    def impute_ratios(self, ptc):
        imp = skimp.IterativeImputer(**self._imputer_args)

        dom_ijd_imputed = imp.fit_transform(ptc.ratios['DoM_ijd'])
        d_ijd_imputed = imp.fit_transform(ptc.ratios['D_ijd'])

        self.fill_nans(ptc.ratios['DoM_ijd'], dom_ijd_imputed)
        self.fill_nans(ptc.ratios['D_ijd'], d_ijd_imputed)
