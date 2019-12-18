import numpy as np
import pandas as pd


# Utility functions

def nanaverage(x, axis=None, weights=None):
    if weights is None:
        return np.nanmean(x)
    notnull = ~(np.isnan(x) | np.isnan(weights))
    return np.average(x[notnull], axis=axis, weights=weights[notnull])


def get_doyr(p):
    """Get ratio between AADT and each daily count."""
    doyr = p.data['Daily Count'].loc[:, ['Date']].copy()
    for year in p.data['AADT'].index:
        doyr.loc[year, 'Day-to-AADT Ratio'] = (
            p.data['AADT'].at[year, 'AADT'] /
            p.data['Daily Count'].loc[year, 'Daily Count']).values
    doyr.reset_index(inplace=True)
    doyr['Month'] = doyr['Date'].dt.month
    doyr['Day of Week'] = doyr['Date'].dt.dayofweek
    return doyr


def get_ndays(doyr):
    """Number of days of each weekday in a year and month."""
    return (
        doyr.reset_index()
        .groupby(['Year', 'Month', 'Day of Week'])['Day-to-AADT Ratio']
        .count().unstack(fill_value=0.))


def get_citywide_growth_factor(rdr, multi_year=False):
    """Citywide growth factor, averaged across all PTCs."""
    return np.mean([v.growth_factor for v in rdr.ptcs.values()
                    if v.data['AADT'].shape[0] > (1 if multi_year else 0)])


def get_neighbours(tc, ptcs, nb, n_neighbours=5, single_direction=True):
    """Find neighbouring PTCs, optionally restricting to same direction."""
    neighbours = nb.get_neighbours(tc.centreline_id)[0]
    if single_direction:
        neighbour_ptcs = [ptcs[n] for n in
                          [tc.direction * nbrs for nbrs in neighbours]
                          if n in ptcs.keys()][:n_neighbours]
    else:
        neighbours = neighbours[:n_neighbours]
        neighbour_ptcs = [ptcs[n] for n in
                          [-nbrs for nbrs in neighbours] + neighbours
                          if n in ptcs.keys()]

    if len(neighbour_ptcs) != (1 if single_direction else 2) * n_neighbours:
        raise ValueError("invalid number of available PTC locations "
                         "for {0}".format(tc.count_id))

    return neighbour_ptcs


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


def aadt_estimator_for_ptcs(rdr, wanted_year,
                            override_growth_factor=False):
    ptc_estimates = []
    for tc in rdr.ptcs.values():
        closest_year = get_closest_year(
            wanted_year, tc.data['AADT'].index.values)
        growth_factor = (override_growth_factor if override_growth_factor
                         else tc.growth_factor)
        aadt_estimate = (
            tc.data['AADT'].loc[closest_year, 'AADT'] *
            growth_factor**(wanted_year - closest_year))

        if np.isnan(aadt_estimate):
            raise ValueError('estimated AADT for {0} is NaN', tc.count_id)

        ptc_estimates.append((tc.count_id, aadt_estimate))

    return pd.DataFrame(ptc_estimates,
                        columns=('Count ID', 'AADT Estimate'))


def combine_estimators(sttc_estimates, ptc_estimates):

    # Outer join estimates together.
    aadt_estimates = pd.merge(sttc_estimates, ptc_estimates,
                              how='outer', on='Count ID',
                              suffixes=('_STTC', '_PTC'))

    # Create a single AADT column that contains both STTC and PTC data.  If
    # both STTC and , take the mean.
    aadt_estimates['AADT'] = aadt_estimates['AADT_STTC']
    aadt_estimates.loc[aadt_estimates['AADT_STTC'].isnull(), 'AADT'] = (
        aadt_estimates.loc[aadt_estimates['AADT_STTC'].isnull(), 'AADT_PTC'])
    bothnotnull = (aadt_estimates['AADT_STTC'].notnull() &
                   aadt_estimates['AADT_PTC'].notnull())
    aadt_estimates.loc[bothnotnull, 'AADT'] = (
        aadt_estimates.loc[bothnotnull, ['AADT_STTC', 'AADT_PTC']]
        .mean(axis=1))

    if aadt_estimates.isnull().any(axis=None):
        raise ValueError("aadt_estimates contains NaNs")

    return aadt_estimates
