import numpy as np
import pandas as pd

from .. import cfg


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


class Matcher:

    def __init__(self, tcs, nb, cfg=cfg.cm):
        self.tcs = tcs
        self.nb = nb
        self.cfg = cfg
        # If ratio matrices have NaNs, prepare annually averaged versions of
        # DoM_ijd and D_ijd.
        if self.any_ptc_ratio_nulls():
            for ptc in self.tcs.ptcs.values():
                self.get_annually_averaged_ratios(ptc)

    def any_ptc_ratio_nulls(self):
        """Check if there are any nulls in PTC ratio matrices."""
        return np.any(
            [self.tcs.ptcs[key].ratios['N_avail_days'].isnull().any(axis=None)
             for key in self.tcs.ptcs.keys()])

    def get_available_years(self, ptc):
        """Calculate a table of years where ratios are available for a given
        month and day of week.

        Only calculated if there are NaNs in PTC derived values.

        Parameters
        ----------
        ptc : permcount.PermCount
            Permanent count.

        Returns
        -------
        avail_years : pd.DataFrame
            Data frame where rows are months, columns are day of week and
            values are lists of available years for each month / day of week.
        """
        avail_years = []
        month = []
        for name, group in (ptc.ratios['N_avail_days']
                            .notnull().groupby('Month')):
            gd = group.reset_index(level='Month', drop=True)
            avail_years.append([gd.loc[gd[c]].index.values
                                for c in group.columns])
            month.append(name)
        return pd.DataFrame(avail_years, index=month)

    def get_annually_averaged_ratios(self, ptc):
        """Annually averaged versions of DoM_ijd and D_ijd.

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
        for i, pyear in enumerate(ptc.perm_years):
            dom_i[i] = nanaverage(
                ptc.ratios['DoM_ijd'].loc[pyear].values,
                weights=ptc.ratios['N_avail_days'].loc[pyear].values)
            # The mean utterly fails in the presence of outliers,
            # so use the **median** (in contravention to TEPs and Bagheri).
            d_i[i] = ptc.ratios['D_ijd'].loc[pyear].unstack().median()

        ptc.ratios['DoM_i'] = pd.DataFrame({'DoM_i': dom_i}, index=idx)
        ptc.ratios['D_i'] = pd.DataFrame({'D_i': d_i}, index=idx)

        ptc.ratios['avail_years'] = self.get_available_years(ptc)

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
        if self.cfg['match_single_direction']:
            # Need to match all neighbours, since not every PTC is
            # bidirectional.
            neighbour_ids = [tc.direction * nbrs for nbrs in neighbours]
        else:
            # Since any direction is valid, can truncate at n_neighbours.
            neighbour_ids = (
                [-nbrs for nbrs in neighbours[:self.cfg['n_neighbours']]] +
                neighbours[:self.cfg['n_neighbours']])
        # Slicing won't do anything for bi-directional matching.
        neighbour_ptcs = [
            self.tcs.ptcs[n] for n in neighbour_ids
            if n in self.tcs.ptcs.keys()][:self.cfg['n_neighbours']]

        # For bi-directional matching, 2 * n_neighbours PTCs are returned.
        neigh_multiplier = 1 if self.cfg['match_single_direction'] else 2
        if len(neighbour_ptcs) != neigh_multiplier * self.cfg['n_neighbours']:
            raise ValueError("invalid number of available PTC locations "
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

    def factor_lookup(self, sttc_row, ptc, default_closest_years):
        closest_year = default_closest_years[sttc_row['Year']]
        loc = ((closest_year, sttc_row['Month']), sttc_row['Day of Week'])

        # Try extracting DoM_ijd.
        dom_ijd = ptc.ratios['DoM_ijd'].at[loc]

        # If not available, find a substitute.
        if np.isnan(dom_ijd):
            avail_years = ptc.ratios['avail_years'].at[
                sttc_row['Month'], sttc_row['Day of Week']]
            # If no year has this month and day of week available, use the
            # annual average.
            if not avail_years.shape[0]:
                dom_ijd = ptc.ratios['DoM_i'].at[closest_year, 'DoM_i']
                d_ijd = ptc.ratios['D_i'].at[closest_year, 'D_i']
                return closest_year, dom_ijd, d_ijd
            # Otherwise pick the closest year that is available, and
            # recalculate loc and dom_ijd
            else:
                closest_year = self.get_closest_year(sttc_row['Year'],
                                                     avail_years)
                loc = ((closest_year, sttc_row['Month']),
                       sttc_row['Day of Week'])
                dom_ijd = ptc.ratios['DoM_ijd'].at[loc]

        # Get d_ijd
        d_ijd = ptc.ratios['D_ijd'].at[loc]
        return closest_year, dom_ijd, d_ijd

    def get_mse_table(self, sttc, ptc):
        """Get a table."""

        # Get date information to match STTC data row with PTC.
        sttc_data = sttc.data.copy()
        sttc_data['Year'] = sttc_data['Date'].dt.year
        sttc_data['Month'] = sttc_data['Date'].dt.month
        sttc_data['Day of Week'] = sttc_data['Date'].dt.dayofweek

        for 