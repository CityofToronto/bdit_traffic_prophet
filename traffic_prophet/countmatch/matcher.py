"""STTC-PTC matcher and AADT estimator classes."""

import numpy as np
import pandas as pd
import sys
from tqdm.auto import tqdm

from .. import cfg


MATCHER_REGISTRY = {}
"""Dict for storing matcher class definitions."""


class MatcherRegistrar(type):
    """Class registry processor, based on `baseband.vdif.header`.

    See https://github.com/mhvk/baseband.

    """

    _registry = MATCHER_REGISTRY

    def __init__(cls, name, bases, dct):

        # Register Matcher subclass if `_matcher_type` not already taken.
        if name not in ('MatcherBase', 'Matcher'):
            if not hasattr(cls, "_matcher_type"):
                raise ValueError("must define a `_matcher_type`.")
            elif cls._matcher_type in MatcherRegistrar._registry:
                raise ValueError("name {0} already registered in "
                                 "MATCHER_REGISTRY".format(cls._matcher_type))

            MatcherRegistrar._registry[cls._matcher_type] = cls

        super().__init__(name, bases, dct)


class Matcher:
    """Base class for matchers."""

    def __new__(cls, matcher_type, *args, **kwargs):
        # __init__ has to be called manually!
        # https://docs.python.org/3/reference/datamodel.html#object.__new__
        # https://stackoverflow.com/questions/20221858/python-new-method-returning-something-other-than-class-instance
        self = super().__new__(MATCHER_REGISTRY[matcher_type])
        self.__init__(*args, **kwargs)
        return self


def nanaverage(x, weights=None):
    """Weighted average of non-null values.

    Parameters
    ----------
    x : array-like
        Array containing data to be averaged.
    weights : array-like or None
        Weights associated with `x`.  If `None`, assumes uniform weights.

    """
    if weights is None:
        return np.nanmean(x)
    notnull = ~(np.isnan(x) | np.isnan(weights))
    return np.average(x[notnull], weights=weights[notnull])


class MatcherBase(metaclass=MatcherRegistrar):

    _average_growth_factor = None

    def __init__(self, tcs, nb, cfg=cfg.cm):
        self.tcs = tcs
        self.nb = nb
        self.cfg = cfg
        # Calculate needed STTC date columns.
        self.get_sttc_date_columns()
        # If ratio matrices have NaNs, prepare annually averaged versions of
        # DoM_ijd and D_ijd as backups.
        self.get_backup_ratios_for_nans()

        if self.cfg['average_growth']:
            self._average_growth_factor = self.get_average_growth()

        self._disable_tqdm = not self.cfg['verbose']

    @property
    def average_growth_factor(self):
        """Mean growth factor averaged over all PTCs."""
        return self._average_growth_factor

    def get_sttc_date_columns(self):
        """Append year, month and day of week columns to STTC data.

        These values are used for matching with neighbouring PTC data.
        """
        for sttc in self.tcs.sttcs.values():
            # This is identical to the 'Year' index, but using it makes
            # `ratio_lookup` ~10-20% faster.
            sttc.data['STTC Year'] = sttc.data['Date'].dt.year
            sttc.data['Month'] = sttc.data['Date'].dt.month
            sttc.data['Day of Week'] = sttc.data['Date'].dt.dayofweek

    def get_backup_ratios_for_nans(self):
        """Get annually averaged ratios for counts with NaNs in ratios."""
        for ptc in self.tcs.ptcs.values():
            if ptc.ratios['N_avail_days'].isnull().any(axis=None):
                self.get_annually_averaged_ratios(ptc)

    def get_annually_averaged_ratios(self, ptc):
        """Annually averaged versions of `DoM_ijd` and `D_ijd`.

        Only useful when there are NaNs in the ratio matrices.

        Parameters
        ----------
        ptc : permcount.PermCount
            Permanent count.

        """
        dom_i = np.ones(ptc.perm_years.shape[0])
        d_i = np.ones(ptc.perm_years.shape[0])
        idx = pd.Index(ptc.perm_years, name='Year')

        # TO DO: consider using the median for DoM_i, as well as for D_i.
        # Will come back to this when resolving #30.
        for i, pyear in enumerate(ptc.perm_years):
            dom_i[i] = nanaverage(
                ptc.ratios['DoM_ijd'].loc[pyear].values,
                weights=ptc.ratios['N_avail_days'].loc[pyear].values)
            # The mean utterly fails in the presence of outliers,
            # so use the **median** (in contravention to TEPs and Bagheri).
            d_i[i] = ptc.ratios['D_ijd'].loc[pyear].unstack().median()

        ptc.ratios['DoM_i'] = pd.DataFrame({'DoM_i': dom_i}, index=idx)
        ptc.ratios['D_i'] = pd.DataFrame({'D_i': d_i}, index=idx)

        ptc.ratios['avail_years'] = self.get_available_years(
            ptc.ratios['N_avail_days'])

    @staticmethod
    def get_available_years(df_N):
        """Calculate a table of permanent years where ratios are available.

        Parameters
        ----------
        df_N : pandas.DataFrame
            DataFrame of the number of available days of each year, month and
            day of week for permanent count years.

        Returns
        -------
        avail_years : pandas.DataFrame
            DataFrame where rows are months, columns are day of week and
            values are lists of available permanent count years for each
            month / day of week.

        """
        # First line turns df_N into index values and single column of
        # number of available days.
        return (df_N.unstack().unstack().reset_index()
                .dropna()
                .groupby(['Month', 'Day of Week'])['Year']
                # pd.Series.unique returns unsorted.  Conversion to list
                # because pandas cannot fill NaNs with arrays.
                .apply(lambda x: sorted(list(x.unique())))
                # `ratio_lookup` checks if values are available using `len()`.
                .unstack(fill_value=[]))

    def get_average_growth(self, multi_year=False):
        """Citywide growth factor, averaged across all PTCs."""
        return np.mean([v.growth_factor for v in self.tcs.ptcs.values()
                        if v.adts['AADT'].shape[0] > (1 if multi_year else 0)])

    def get_neighbour_ptcs(self, tc):
        """Find neighbouring PTCs.

        The number of neighbours to return, and whether matching is restricted
        only to PTCs in the same direction, is set in self.cfg.

        Parameters
        ----------
        tc : base.Count subclass
            (Short-term) count.

        """
        neighbours = self.nb.get_neighbours(tc.centreline_id)[0]

        # Need to match all neighbours, since not every PTC is
        # bidirectional.
        neighbour_ids = [tc.direction * nbrs for nbrs in neighbours]
        if not self.cfg['match_single_direction']:
            # Interleave same and opposite direction counts (with a preference
            # for same direction).  Not all of these will be PTCs, hence the
            # next step below.
            neighbour_ids = [cid for cid_sd in neighbour_ids
                             for cid in (cid_sd, -cid_sd)]

        neighbour_ptcs = [
            self.tcs.ptcs[n] for n in neighbour_ids
            if n in self.tcs.ptcs.keys()][:self.cfg['n_neighbours']]

        if len(neighbour_ptcs) < self.cfg['n_neighbours']:
            raise ValueError("too few available PTC locations "
                             "for {0}".format(tc.count_id))

        return neighbour_ptcs

    @staticmethod
    def get_closest_year(sttc_years, ptc_years):
        """Find closest year to an STTC count available at a PTC location."""
        if isinstance(sttc_years, np.ndarray):
            # Outer product to determine absolute difference between
            # STTC years and PTC years.
            mindiff_arg = np.argmin(abs(sttc_years[:, np.newaxis] - ptc_years),
                                    axis=1)
            return ptc_years[mindiff_arg]
        # If sttc_years is a single value, can just do a standard argmin.
        return ptc_years[np.argmin(abs(sttc_years - ptc_years))]

    def ratio_lookup(self, sttc_row, ptc, default_closest_years):
        """Find closest match `DoM_ijd` and `D_ijd` ratios for STTC row."""
        # Try extracting DoM_ijd from closest year and same month/DoW.
        closest_year = default_closest_years[sttc_row['STTC Year']]
        loc = ((closest_year, sttc_row['Month']), sttc_row['Day of Week'])
        dom_ijd = ptc.ratios['DoM_ijd'].at[loc]

        # If this is not available, find a substitute.
        if np.isnan(dom_ijd):
            avail_years = ptc.ratios['avail_years'].at[
                sttc_row['Month'], sttc_row['Day of Week']]
            # If no year has this month and day of week available, use the
            # annual average.
            if not len(avail_years):
                dom_ijd = ptc.ratios['DoM_i'].at[closest_year, 'DoM_i']
                d_ijd = ptc.ratios['D_i'].at[closest_year, 'D_i']

                return closest_year, dom_ijd, d_ijd

            # Otherwise pick the closest year that is available, and
            # recalculate loc and dom_ijd
            else:
                # avail_years is a list; can make this faster by
                # get_avail_years to return arrays instead of non-zero-length
                # lists, but that would lead to mixed data types.
                closest_year = self.get_closest_year(sttc_row['STTC Year'],
                                                     np.array(avail_years))
                loc = ((closest_year, sttc_row['Month']),
                       sttc_row['Day of Week'])
                dom_ijd = ptc.ratios['DoM_ijd'].at[loc]

        # Get d_ijd as well.
        d_ijd = ptc.ratios['D_ijd'].at[loc]
        return closest_year, dom_ijd, d_ijd

    def get_ratio_from_ptc(self, sttc, ptc, n_switch_to_merge=80):
        """Get a table of PTC ratios for estimating STTC monthly pattern.

        For larger amounts of data, processing unique times from `sttc.data`
        then merging with it is faster than processing every row.

        Parameters
        ----------
        sttc : base.Count subclass
            Short-term count.
        ptc : permcount.PermCount
            Permanent count.
        n_switch_to_merge : int
            Number of rows in `sttc.data` to switch from row-by-row calculation
            to processing unique times then merging.  For performance testing.

        """
        use_merge = sttc.data.shape[0] > n_switch_to_merge

        sttc_timeinfo = sttc.data[['STTC Year', 'Month', 'Day of Week']]
        if use_merge:
            # Obtain a unique year, month and day of week table.
            ijd = sttc_timeinfo.drop_duplicates().reset_index(drop=True)
        else:
            # ijd isn't modified, so might as well use `sttc_timeinfo` (which
            # is a view).
            ijd = sttc_timeinfo

        # Get a lookup table assuming all PTC permanent years are available.
        sttc_years = ijd['STTC Year'].unique()
        default_closest_years = dict(zip(
            sttc_years, self.get_closest_year(sttc_years, ptc.perm_years)))

        # Row-by-row search for matching PTC values.
        ptc_vals = [self.ratio_lookup(row, ptc, default_closest_years)
                    for _, row in ijd.iterrows()]
        # `index=ijd.index` allows pd.concat to work properly when merging
        # isn't used.
        ptc_vals = pd.DataFrame(
            ptc_vals, index=ijd.index,
            columns=['Closest PTC Year', 'DoM_ijd', 'D_ijd'])
        ratios = pd.concat([ijd, ptc_vals], axis=1)

        if use_merge:
            ratios = (pd.merge(sttc_timeinfo, ratios, how='left',
                               left_on=('STTC Year', 'Month', 'Day of Week'),
                               right_on=('STTC Year', 'Month', 'Day of Week'))
                      .set_index(sttc_timeinfo.index))

        return ratios

    def get_monthly_pattern(self, sttc, ptc, want_year):
        """Get a table of MADT and AADT estimates using matched PTC ratios.

        Parameters
        ----------
        sttc : base.Count subclass
            Short-term count.
        ptc : permcount.PermCount
            Permanent count.

        """
        mvals = self.get_ratio_from_ptc(sttc, ptc)
        # Python is beautiful: self.average_growth_factor doesn't need to be
        # defined unless self.cfg['average_growth'] is False.
        growth_factor = (self.average_growth_factor
                         if self.cfg['average_growth'] else ptc.growth_factor)

        if not np.array_equal(sttc.data.index.values, mvals.index.values):
            raise ValueError("sttc and mvals indices don't match!")

        # Eqn. 2 Bagheri.
        year_diff = want_year - mvals['STTC Year'].values
        mvals['MADT_est'] = (
            sttc.data['Daily Count'].values * mvals['DoM_ijd'].values *
            growth_factor**year_diff)

        mpattern = pd.DataFrame(
            {'MADT_est': mvals.groupby('Month')['MADT_est'].mean()})

        # Eqn. 3 Bagheri.
        mpattern['AADT_est'] = (
            sttc.data['Daily Count'].values * mvals['D_ijd'].values *
            growth_factor**year_diff).mean()

        # Eqn. 5 Bagheri.
        mpattern['MF_STTC'] = (mpattern['MADT_est'].values /
                               mpattern['AADT_est'].values)

        # Don't bother storing values already in other data frames (except
        # 'STTC_Year', since it's required by the AADT estimator).
        mvals.drop(columns=['Month', 'Day of Week'], inplace=True)

        return {'Match Values': mvals, 'Growth Factor': growth_factor,
                'Monthly Pattern': mpattern}

    def estimate_mse(self, mpattern, ptc, want_year):
        """Estimate MSE between PTC and STTC monthly pattern.

        Parameters
        ----------
        mpattern : pandas.DataFrame
            'Monthly Pattern' output of `get_monthly_pattern`.
        ptc : permcount.PermCount
            Permanent count.
        want_year : int
            Year of analysis.

        """
        ptc_closest_year = self.get_closest_year(want_year, ptc.perm_years)
        # No need to use a growth factor, since it's canceled out by the
        # normalization.
        mf_ptc = pd.Series(
            ptc.adts['MADT'].loc[ptc_closest_year, 'MADT'] /
            ptc.adts['AADT'].loc[ptc_closest_year, 'AADT'])

        # `mean` skips NaNs.
        mse = ((mpattern['MF_STTC'] - mf_ptc)**2).mean()
        if mse < sys.float_info.epsilon:
            mse = 0.

        return mse

    def get_mmse_aadt(self, tc_data, mmse_mvals, mmse_growth_factor,
                      want_year):
        """Obtain AADT based on minimum MSE match.

        Parameters
        ----------
        tc_data : pandas.DataFrame
            STTC daily counts.
        mmse_mvals : pandas.DataFrame
            'Match Values' from `get_monthly_pattern` with MSE matching PTC.
        mmse_growth_factor : float
            Growth factor.
        want_year : int
            Year of analysis.

        """
        # TO DO: there are definitely cases where years before the closest year
        # have way more data - surely there's a better way of doing this?
        mmse_closest_year = self.get_closest_year(
            want_year, mmse_mvals['STTC Year'].unique())
        mmse_mvals_cy = mmse_mvals.loc[mmse_closest_year]
        mmse_counts_cy = tc_data.loc[mmse_closest_year]

        if not np.array_equal(mmse_mvals_cy.index.values,
                              mmse_counts_cy.index.values):
            raise ValueError("sttc_counts and mmse_mvals indices don't match!")

        aadt_est = (mmse_counts_cy['Daily Count'].values *
                    mmse_mvals_cy['D_ijd'].values *
                    mmse_growth_factor**(want_year - mmse_closest_year)).mean()

        return aadt_est

    def estimate_sttc_aadt(self, tc, want_year):
        # Only implemented in subclasses.
        raise NotImplementedError

    def estimate_ptc_aadt(self, ptc, want_year):
        """Estimates AADT of a PTC.

        Parameters
        ----------
        ptc : permcount.PermCount
            Permanent count.
        want_year : int
            Year of analysis.

        """
        closest_year = self.get_closest_year(want_year, ptc.perm_years)
        growth_factor = (self.average_growth_factor
                         if self.cfg['average_growth'] else
                         ptc.growth_factor)
        aadt_estimate = (ptc.adts['AADT'].loc[closest_year, 'AADT'] *
                         growth_factor**(want_year - closest_year))
        return aadt_estimate

    def estimate_aadts(self, want_year):
        """Estimates AADTs for all PTCs and STTCs.

        Parameters
        ----------
        want_year : int
            Year of analysis.

        """
        sttc_aadt_ests = []
        for tc in tqdm(self.tcs.sttcs.values(),
                       desc='Estimating STTC AADTs',
                       disable=self._disable_tqdm):
            sttc_aadt_ests.append(
                (tc.count_id, self.estimate_sttc_aadt(tc, want_year)))

        # Process PTC AADT estimates.
        ptc_aadt_ests = []
        for ptc in tqdm(self.tcs.ptcs.values(),
                        desc='Estimating PTC AADTs',
                        disable=self._disable_tqdm):
            ptc_aadt_ests.append(
                (ptc.count_id, self.estimate_ptc_aadt(ptc, want_year)))

        sttc_aadt_ests = (pd.DataFrame(
            sttc_aadt_ests, columns=('Count ID', 'AADT Estimate'))
            .sort_values(by='Count ID').reset_index(drop=True))

        ptc_aadt_ests = (pd.DataFrame(
            ptc_aadt_ests, columns=('Count ID', 'AADT Estimate'))
            .sort_values(by='Count ID').reset_index(drop=True))

        return sttc_aadt_ests, ptc_aadt_ests


class MatcherStandard(MatcherBase):
    """TEPs-based matcher.

    When calculating the MSE between `MF_STTC` and `MF_PTC` from a nearby PTC,
    uses ratios from that PTC.

    Parameters
    ---------------
    tcs : countmatch.reader subclass
        Reader object containing PTCs and STTCs already processed by
        `derivedvals` methods.
    nb : countmatch.neighbour.NeighbourBase subclass
        Nearest neighbour lookup class.
    cfg : dict, optional
        Settings and hyperparameters.  Default: `config.cm`.

    """

    _matcher_type = 'Standard'

    def estimate_sttc_aadt(self, tc, want_year):
        """Estimates AADT of an STTC

        Parameters
        ----------
        tc : base.Count
            Short-term count.
        want_year : int
            Year of analysis.

        """

        neighbour_ptcs = self.get_neighbour_ptcs(tc)

        # Store monthly patterns and mean square errors of comparisons with
        # neighbouring PTCs.
        tc.mpatterns = {}
        mses = []
        for ptc in neighbour_ptcs:
            tc.mpatterns[ptc.count_id] = self.get_monthly_pattern(tc, ptc,
                                                                  want_year)
            mses.append((ptc.count_id, self.estimate_mse(
                tc.mpatterns[ptc.count_id]['Monthly Pattern'], ptc, want_year))
            )
        tc.mses = pd.DataFrame(mses, columns=['Count ID', 'MSE'])

        # Find the smallest MSE (in case of ties, first index (closer distance)
        # is chosen).  A bit clunkier than setting the index to 'Count ID' to
        # keep its design closer to MatcherBagheri.
        mmse_count_id = tc.mses.at[tc.mses['MSE'].idxmin(), 'Count ID']

        aadt_est = self.get_mmse_aadt(
            tc.data, tc.mpatterns[mmse_count_id]['Match Values'],
            tc.mpatterns[mmse_count_id]['Growth Factor'], want_year)

        return aadt_est


class MatcherBagheri(MatcherBase):
    """Bagheri-based matcher.

    When calculating the MSE between `MF_STTC` and `MF_PTC` from a nearby PTC,
    uses ratios from the closest PTC (there are too few PTCs to use only ones
    of the same road class as the STTC).

    Parameters
    ---------------
    tcs : countmatch.reader subclass
        Reader object containing PTCs and STTCs already processed by
        `derivedvals` methods.
    nb : countmatch.neighbour.NeighbourBase subclass
        Nearest neighbour lookup class.
    err_measure : str, optional
        Measure of error, either 'MSE' or 'COV'.  Default: 'MSE'.
    cfg : dict, optional
        Settings and hyperparameters.  Default: `config.cm`.

    """
    _matcher_type = 'Bagheri'

    def __init__(self, tcs, nb, err_measure='MSE', cfg=cfg.cm):
        assert err_measure in ('MSE', 'COV'), "unrecognized err_measure!"
        if err_measure == 'COV':
            self._err_func = self.estimate_cov
        else:
            self._err_func = self.estimate_mse
        super().__init__(tcs, nb, cfg=cfg)

    def estimate_cov(self, mpattern, ptc, want_year):
        """Estimate COV between PTC and STTC monthly pattern.

        Parameters
        ----------
        mpattern : pandas.DataFrame
            'Monthly Pattern' output of `get_monthly_pattern`.
        ptc : permcount.PermCount
            Permanent count.
        want_year : int
            Year of analysis.

        """
        ptc_closest_year = self.get_closest_year(want_year, ptc.perm_years)
        ptc_madt = ptc.adts['MADT'].loc[ptc_closest_year, 'MADT']
        ratio = mpattern['MADT_est'] / ptc_madt
        # No need to use a growth factor, since it's canceled out in the ratio.
        cov = ratio.std() / ratio.mean()
        if cov < sys.float_info.epsilon:
            cov = 0.
        return cov

    def estimate_sttc_aadt(self, tc, want_year):
        """Estimates AADT of an STTC

        Parameters
        ----------
        tc : base.Count
            Short-term count.
        want_year : int
            Year of analysis.

        """
        neighbour_ptcs = self.get_neighbour_ptcs(tc)

        # Store monthly patterns and mean square errors of comparisons with
        # neighbouring PTCs.  Unlike with `MatcherStandard`, we'll only store
        # nearest (in distance) and best match patterns.
        baseline_mpattern = self.get_monthly_pattern(tc, neighbour_ptcs[0],
                                                     want_year)
        tc.mpatterns = {neighbour_ptcs[0].count_id: baseline_mpattern}

        mses = []
        for ptc in neighbour_ptcs:
            mses.append((ptc.count_id, self._err_func(
                baseline_mpattern['Monthly Pattern'], ptc, want_year)))
        tc.mses = pd.DataFrame(mses, columns=['Count ID', 'MSE'])

        # Find the smallest MSE (in case of ties, first index (closer distance)
        # is chosen).
        mmse_id = tc.mses['MSE'].idxmin()

        if mmse_id > 0:
            best_mpattern = self.get_monthly_pattern(
                tc, neighbour_ptcs[mmse_id], want_year)
            # Save to `tc.mpatterns` for diagonstic purposes.
            tc.mpatterns[neighbour_ptcs[mmse_id].count_id] = best_mpattern
        else:
            best_mpattern = baseline_mpattern
        mmse_mvals = best_mpattern['Match Values']
        mmse_growth_factor = best_mpattern['Growth Factor']

        aadt_est = self.get_mmse_aadt(tc.data, mmse_mvals, mmse_growth_factor,
                                      want_year)

        return aadt_est


def match(tcs, nb, want_year, cfg=cfg.cm):
    matcher = Matcher(cfg['matcher'], tcs, nb,
                      cfg=cfg, **cfg['matcher_settings'])
    return matcher.estimate_aadts(want_year)
