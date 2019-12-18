import numpy as np
import pandas as pd
import sys

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
    N_days = cmc.get_ndays(doyr)

    dom_avg = pd.DataFrame(
        {'DoM_i': np.empty(ptc.data['AADT'].shape[0])},
        index=ptc.data['AADT'].index)

    for year in dom_avg.index:
        dom_avg.loc[year, 'DoM_i'] = (
            cmc.nanaverage(ptc.data['DoM Factor'].loc[year, :].values,
                           weights=N_days.loc[year, :].values))

    ptc.data['DoM_i'] = dom_avg


def get_available_years(ptc):
    avail_years = []
    month = []

    for name, group in ptc.data['D_ijd'].notnull().groupby(level=1):
        gd = group.reset_index(level=1, drop=True)
        avail_years.append([gd.loc[gd[c]].index.values for c in group.columns])
        month.append(name)

    ptc.data['D_ijd_avail'] = pd.DataFrame(avail_years, index=month)


def get_unstacked_factors(ptc):
    ptc.data['factors_unstacked'] = pd.DataFrame({
        'D_ijd': ptc.data['D_ijd'].stack(),
        'DoM_ijd': ptc.data['DoM Factor'].stack().sort_values()})
    ptc.data['factors_unstacked'].reset_index(inplace=True)


def pinpoint_factor_lookup(sttc_row, ptc):
    # Check the year availability for the particular month and day of week
    # (D_ijd and DoM_ijd should have the same availability)
    dijd_avail_check = ptc.data['D_ijd_avail'].at[sttc_row['Month'],
                                                  sttc_row['Day of Week']]
    # If available, get the closest year with month and day of week.
    if dijd_avail_check.shape[0]:
        closest_year = cmc.get_closest_year(sttc_row['Year'], dijd_avail_check)
        dijd = ptc.data['D_ijd'].at[
            (closest_year, sttc_row['Month']), sttc_row['Day of Week']]
        domijd = ptc.data['DoM Factor'].at[
            (closest_year, sttc_row['Month']), sttc_row['Day of Week']]
    # If not available, use the annual average from the closest year.
    else:
        closest_year = cmc.get_closest_year(
            sttc_row['Year'], ptc.data['AADT'].index.values)
        dijd = ptc.data['D_i'].at[closest_year, 'D_i']
        domijd = ptc.data['DoM_i'].at[closest_year, 'DoM_i']
    return closest_year, dijd, domijd


def get_factors_from_ptc_pinpoint(sttc, ptc):
    # As get_factors_from_ptc, but use a point-by-point lookup.

    # Unstack daily count table.
    daily_count = sttc.data.reset_index().drop(columns='Day of Year')
    daily_count['Day of Week'] = daily_count['Date'].dt.dayofweek
    daily_count['Month'] = daily_count['Date'].dt.month

    # Obtain unique year, month and day of week table.
    unique_ijd = (daily_count[['Year', 'Month', 'Day of Week']]
                  .drop_duplicates())

    # Row-by-row search for closest D_ijd and DoM_ijd.
    closest_year_arr = []
    d_ijd_arr = []
    dom_ijd_arr = []    

    for i, row in unique_ijd.iterrows():
        closest_year, d_ijd, dom_ijd = pinpoint_factor_lookup(row, ptc)
        closest_year_arr.append(closest_year)
        d_ijd_arr.append(d_ijd)
        dom_ijd_arr.append(dom_ijd)

    unique_ijd['Closest Year'] = closest_year_arr
    unique_ijd['D_ijd'] = d_ijd_arr
    unique_ijd['DoM_ijd'] = dom_ijd_arr

    # Merge back with daily counts to assign each row a set of scaling factors.
    return pd.merge(daily_count, unique_ijd,
                    left_on=('Year', 'Month', 'Day of Week'),
                    right_on=('Year', 'Month', 'Day of Week'))


def get_closest_year_table(ptc_match, ptc):
    sttc_years = ptc_match['Year'].unique()
    ptc_years = ptc.data['AADT'].index.values

    return pd.DataFrame(
        {'Year': sttc_years,
         'Closest Year': cmc.get_closest_year(sttc_years, ptc_years)})


def fix_factors_from_ptc_merge(ptc_match, sttc, ptc):

    nan_idxs = ptc_match[ptc_match['D_ijd'].isnull()].index

    for i in nan_idxs:
        (ptc_match.loc[i, 'Closest Year'], ptc_match.loc[i, 'D_ijd'],
         ptc_match.loc[i, 'DoM_ijd']) = pinpoint_factor_lookup(
             ptc_match.loc[i, :], ptc)


def get_factors_from_ptc_merge(sttc, ptc):

    # Unstack daily count table.
    daily_count = sttc.data.reset_index().drop(columns='Day of Year')
    daily_count['Day of Week'] = daily_count['Date'].dt.dayofweek
    daily_count['Month'] = daily_count['Date'].dt.month

    # Obtain unique year, month and day of week table.
    unique_ijd = (daily_count[['Year', 'Month', 'Day of Week']]
                  .drop_duplicates())
    # For each row, get the closest year available in the PTC data.
    unique_ijd = pd.merge(unique_ijd, get_closest_year_table(unique_ijd, ptc),
                          how='left', left_on='Year', right_on='Year')
    # Obtain D_ijd and DoM_ijd from closest year, same month and day for each
    # ijd value.
    unique_ijd = pd.merge(
        unique_ijd, ptc.data['factors_unstacked'], how='left',
        left_on=('Closest Year', 'Month', 'Day of Week'),
        right_on=('Year', 'Month', 'Day of Week'),
        suffixes=('', '_r'))
    unique_ijd.drop(columns='Year_r', inplace=True)

    # Merge back with daily counts to assign each row a set of scaling factors.
    ptc_match = pd.merge(daily_count, unique_ijd,
                         left_on=('Year', 'Month', 'Day of Week'),
                         right_on=('Year', 'Month', 'Day of Week'))

    # Fix any NaNs that crop up because
    return fix_factors_from_ptc_merge(ptc_match, sttc, ptc)


def get_ptc_match(sttc, ptc):
    if sttc.data.shape[0] >= 50:
        ptc_match = get_factors_from_ptc_merge(sttc, ptc)
    else:
        ptc_match = get_factors_from_ptc_pinpoint(sttc, ptc)
    if ptc_match.isnull().any(axis=None):
        raise ValueError('found a NaN for', sttc.count_id)
    return ptc_match


def estimate_mse(ptc_match, sttc, ptc, wanted_year):
    ptc_match['MADT_est'] = (
        ptc_match['Daily Count'] * ptc_match['DoM_ijd'] *
        ptc.growth_factor**(wanted_year - ptc_match['Year']))
    madt_est = pd.DataFrame(
        {'MADT_estimate': ptc_match.groupby('Month')['MADT_est'].mean()})
    madt_est['AADT_est'] = (
        ptc_match['Daily Count'] * ptc_match['D_ijd'] *
        ptc.growth_factor**(wanted_year - ptc_match['Year'])).mean()
    madt_est['MF_STTC'] = madt_est['MADT_estimate'] / madt_est['AADT_est']

    ptc_closest_year = cmc.get_closest_year(
        wanted_year, ptc.data['AADT'].index.values)
    madt_est['MF_PTC'] = (ptc.data['MADT'].loc[ptc_closest_year, 'MADT'] /
                          ptc.data['AADT'].loc[ptc_closest_year, 'AADT'])

    mse = np.mean((madt_est['MF_STTC'] - madt_est['MF_PTC'])**2)
    if mse < sys.float_info.epsilon:
        mse = 0.

    return mse, madt_est


def get_aadt_estimate_for_sttc(tc, ptcs, nb, wanted_year):

    # Find nearest neighbours in same direction.
    neighbour_ptcs = cmc.get_neighbours(
        tc, ptcs, nb, n_neighbours=n_neighbours,
        single_direction=single_direction)
    
    # Determine minimum MSE count of five closest neighbouring PTCs.
    mses = []
    for ptc in neighbour_ptcs:
        mses.append(estimate_mse(tc, ptc, wanted_year))
    mmse_ptc_match = min(mses, key=lambda x: x[0])[1]

    # Estimate AADT using most recent year of STTC counts and
    # MADT pattern of closest PTC.
    closest_year = get_closest_year(
        wanted_year, mmse_ptc_match['Year'].unique())
    mmse_ptc_match_cy = mmse_ptc_match.loc[
        mmse_ptc_match['Year'] == closest_year, :]
    aadt_est_closest_year = (
        mmse_ptc_match_cy['Daily Count'] * mmse_ptc_match_cy['D_ijd'] *
        ptc.growth_factor**(wanted_year - mmse_ptc_match_cy['Year'])).mean()

    if np.isnan(aadt_est_closest_year):
        raise ValueError('Found a NaN in outer loop', tc.count_id)

    tc.aadt_estimate = aadt_est_closest_year


def countmatch(rdr):
    # Preprocess PTCs.
    for ptc in rdr.ptcs.values():
        get_Dijd(ptc)
        get_DoMi(ptc)
        get_available_years(ptc)
        get_unstacked_factors(ptc)
    
    ptc_match = get_ptc_match(sttc, ptc)
