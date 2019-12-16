import numpy as np
import pandas as pd

import countmatch_common as cmc


def get_Dijd(ptc):
    # Get ratio between daily count and AADT, grouping by month and day of week
    doyadt = []
    for year in ptc.data['AADT'].index:
        _ctable = (ptc.data['AADT'].at[year, 'AADT'] /
                   ptc.data['DoMADT'].loc[year])
        _ctable.index = pd.MultiIndex.from_product(
            [[year, ], _ctable.index],
            names=['Year', _ctable.index.name])
        doyadt.append(_ctable)
    ptc.data['D_ijd'] = pd.concat(doyadt)

    # Get ratio between AADT and daily count
    doyr = pd.DataFrame(
        {'D_i': np.empty(ptc.data['AADT'].shape[0])},
        index=ptc.data['AADT'].index)
    for year in doyr.index.values:
        # The mean utterly fails in the presence of outliers,
        # so use the **median** (in contravention to TEPs and Bagheri)
        doyr.loc[year, 'D_i'] = (
            ptc.data['AADT'].loc[year, 'AADT'] /
            ptc.data['Daily Count'].loc[year, 'Daily Count']).median()

    ptc.data['D_i'] = doyr


def get_DoMi(ptc):
    doyr = cmc.get_doyr(ptc)

    N_days = (doyr.reset_index()
              .groupby(['Year', 'Month', 'Day of Week'])['Day-to-AADT Ratio']
              .count().unstack(fill_value=0.))

    dom_avg = pd.DataFrame(
        {'DoM_i': np.empty(ptc.data['AADT'].shape[0])},
        index=ptc.data['AADT'].index)

    for year in dom_avg.index:
        weights = N_days.loc[year, :]
        dom_avg.loc[year, 'DoM_i'] = (
            nanaverage(ptc.data['DoM Factor'].loc[year, :].values,
                       weights=N_days.loc[year, :].values))

    ptc.data['DoM_i'] = dom_avg


def countmatch(rdr):
    for ptc in rdr.ptcs.values():
        get_Dijd(ptc)