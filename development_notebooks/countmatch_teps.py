import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import countmatch_common as cmc


def mse_preprocess_ptc(p):
    # Get ratio between AADT and each daily count.
    doyr = cmc.get_doyr(p)
    
    # Number of days of the week in each year and month.  Fill NaNs with
    # 0.
    N_days = cmc.get_ndays(doyr)

    # Create two arrays - first breaks down values by day-of-week and year.
    # First, get day-to-AADT ratios for each day of week and year.
    ptc_mse_ydow = pd.DataFrame(
        doyr.groupby(['Day of Week', 'Year'])['Day-to-AADT Ratio'].mean())

    # Then, for each year and day-of-week, calculate MADT, DoMADT, etc.
    # averages over all months.
    madt_avg = []
    dom_avg = []
    domadt_avg = []
    for dow, year in ptc_mse_ydow.index:
        # Double loc is shortest way I've discovered to get single-index
        # series.
        weights = N_days.loc[year, :].loc[:, dow]

        madt_avg.append(np.average(
            p.data['MADT'].loc[year, 'MADT'][weights.index].values,
            weights=weights.values))
        dom_avg.append(cmc.nanaverage(
            p.data['DoM Factor'].loc[year, :].loc[weights.index, dow],
            weights=weights.values))
        domadt_avg.append(cmc.nanaverage(
            p.data['DoMADT'].loc[year, :].loc[weights.index, dow],
            weights=weights.values))

    ptc_mse_ydow['MADT Avg.'] = madt_avg
    ptc_mse_ydow['DoM Factor Avg.'] = dom_avg
    ptc_mse_ydow['DoMADT Avg.'] = domadt_avg

    if (ptc_mse_ydow['MADT Avg.'].isnull().values.any() or
            ptc_mse_ydow['DoM Factor Avg.'].isnull().values.any() or
            ptc_mse_ydow['DoM Factor Avg.'].isnull().values.any()):
        raise ValueError("Weighted monthly averages for {0} "
                         "resulted in NaNs.".format(p.count_id))

    # Now, create a second array that averages over both month and day-of-week.
    ptc_mse_y = pd.DataFrame(doyr.groupby('Year')['Day-to-AADT Ratio'].mean())

    madt_avg = []
    dom_avg = []
    domadt_avg = []
    for year in ptc_mse_y.index:
        n_days_year = N_days.loc[year, :]
        madt_weights = n_days_year.sum(axis=1, skipna=True)

        madt_avg.append(np.average(
            p.data['MADT'].loc[year, 'MADT'][madt_weights.index].values,
            weights=madt_weights.values))

        dom_year = p.data['DoM Factor'].loc[year, :]
        dom_avg.append(cmc.nanaverage(
            dom_year.values, weights=n_days_year.values))
        domadt_avg.append(cmc.nanaverage(
            p.data['DoMADT'].loc[year, :].values, weights=n_days_year.values))

    ptc_mse_y['MADT Avg.'] = madt_avg
    ptc_mse_y['DoM Factor Avg.'] = dom_avg
    ptc_mse_y['DoMADT Avg.'] = domadt_avg

    if (ptc_mse_y['MADT Avg.'].isnull().values.any() or
            ptc_mse_y['DoM Factor Avg.'].isnull().values.any() or
            ptc_mse_y['DoM Factor Avg.'].isnull().values.any()):
        raise ValueError("Weighted annual averages for {0} "
                         "resulted in NaNs.".format(p.count_id))

    p.data['Day-to-AADT Factors'] = doyr
    p.data['MSE Annual Averages'] = ptc_mse_y
    p.data['MSE Annual-DoW Averages'] = ptc_mse_ydow


def get_normalized_seasonal_patterns(tc, ptcs, nb, want_year,
                                     single_direction=True):
    """Get the normalized seasonal pattern for a count.

    For STTCs, get best estimate normalized patterns and corresponding PTC
    normalized patterns to check for MSE (Eqn. 6 in Bagheri).  For PTCs, get
    best estimate from nearby PTCs and check as a part of validation.
    """

    # Find neighbouring PTCs by first finding neighbouring centreline IDs,
    # then checking if either direction exists in rptcs.
    neighbour_ptcs = get_neighbours(tc, ptcs, nb,
                                    single_direction=single_direction)

    # Declare the columns in the final saved data frame.
    tc_msedata = []

    if tc.is_permanent:
        tc_dc = tc.data['Daily Count'].reset_index()
    else:
        tc_dc = tc.data.reset_index()
    tc_dc['Day of Week'] = tc_dc['Date'].dt.dayofweek

    for i, row in tc_dc.iterrows():
        ryear, rdow = row['Year'], row['Day of Week']

        for p in neighbour_ptcs:

            if rdow in p.mse_pp['MSE_yDoW'].index.levels[0]:
                unique_years = p.mse_pp['MSE_yDoW'].loc[rdow].index.values
                closest_year = unique_years[np.argmin(
                    np.abs(unique_years - ryear))]

                (day_to_aadt_ratio_avg, madt_avg, dom_avg, domadt_avg) = (
                    p.mse_pp['MSE_yDoW'].loc[(rdow, closest_year)])
            else:
                # Levels contain all unique years regardless if each is
                # available for every day-of-week.
                unique_years = p.mse_pp['MSE_yDoW'].index.levels[1].values
                closest_year = unique_years[np.argmin(
                    np.abs(unique_years - ryear))]

                (day_to_aadt_ratio_avg, madt_avg, dom_avg, domadt_avg) = (
                    p.mse_pp['MSE_y'].loc[closest_year])
            
            if madt_avg is np.nan:
                raise ValueError("ummmmm, this can't be NaN.")

            aadt_closest_year = p.data['AADT'].at[closest_year, 'AADT']

            tc_year.append(ryear)
            tc_dayofyear.append(row['Day of Year'])
            tc_ptcid.append(p.count_id)
            tc_day_to_aadt_ratio_avg.append(day_to_aadt_ratio_avg)
            tc_madt_avg.append(madt_avg)
            tc_dom_avg.append(dom_avg)
            tc_domadt_avg.append(domadt_avg)
            tc_closest_year.append(closest_year)
            tc_aadt_closest_year.append(aadt_closest_year)

    tc_mse = pd.DataFrame({
        'Year': tc_year,
        'Day of Year': tc_dayofyear,
        'PTC ID': tc_ptcid,
        'PTC Day-to-AADT Ratio': tc_day_to_aadt_ratio_avg,
        'PTC MADT Avg.': tc_madt_avg,
        'PTC DoM Factor Avg.': tc_dom_avg,
        'PTC DoMADT Avg.': tc_domadt_avg,
        'PTC Closest Year AADT': tc_aadt_closest_year
    })

    tc_mse = pd.merge(tc_dc, tc_mse, on=('Year', 'Day of Year'))

    # I disagree with this, but line 95 of DoMSTTC.m seems to do it.
    mean_tc_count = tc_dc['Daily Count'].mean()

    tc_mse['AADT_prelim'] = (
        mean_tc_count * tc_mse['PTC Day-to-AADT Ratio'] *
        growth_rate_citywide**(want_year - tc_mse['Year']))
    tc_mse['MADT_pj'] = (
        tc_mse['Daily Count'] * tc_mse['PTC DoM Factor Avg.'] *
        growth_rate_citywide**(want_year - tc_mse['Year']))
    tc_mse['MF_STTC'] = tc_mse['MADT_pj'] / tc_mse['AADT_prelim']
    tc_mse['MF_PTC'] = (tc_mse['PTC MADT Avg.'] /
                        tc_mse['PTC Closest Year AADT'])

    tc.tc_mse = tc_mse



def estimate_aadts(rdr, nb, want_year, progress_bar=False):

    for p in tqdm(rdr.ptcs.values(),
                  desc='Calculating PTC annual/DoW averages',
                  disable=(not progress_bar)):
        mse_preprocess_ptc(p)

    citywide_growth_factor = cmc.get_citywide_growth_factor(rdr)

    for tc in tqdm(rdr.sttcs.values(),
                   desc='Calculating STTC normalized monthly patterns',
                   disable=(not progress_bar)):