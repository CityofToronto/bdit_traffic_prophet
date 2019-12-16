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


def get_neighbours(tc, ptcs, nb, single_direction=True):
    """Find neighbouring PTCs, optionally restricting to same direction."""
    neighbours = nb.get_neighbours(tc.centreline_id)[0]
    if single_direction:
        neighbour_ptcs = [ptcs[n] for n in
                          [-nbrs for nbrs in neighbours] + neighbours
                          if n in ptcs.keys()][:10]

        if len(neighbour_ptcs) != 10:
            raise ValueError("invalid number of available PTC locations "
                             "for {0}".format(tc.count_id))

    else:
        neighbour_ptcs = [ptcs[n] for n in
                          [tc.direction * nbrs for nbrs in neighbours]
                          if n in ptcs.keys()][:5]

        if len(neighbour_ptcs) != 5:
            raise ValueError("invalid number of available PTC locations "
                             "for {0}".format(tc.count_id))

    return neighbour_ptcs
