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


def get_normalized_seasonal_patterns(
        tc, ptcs, nb, want_year, n_neighbours=5, single_direction=True):
    """Get the normalized seasonal pattern for a count."""

    # Find neighbouring PTCs by first finding neighbouring centreline IDs,
    # then checking if either direction exists in rptcs.
    neighbour_ptcs = cmc.get_neighbours(
        tc, ptcs, nb, n_neighbours=n_neighbours,
        single_direction=single_direction)

    # Declare the columns in the final saved data frame.
    tc_msedata = []

    if tc.is_permanent:
        # We shouldn't be allowing PTCs to be passed into this right now.
        # tc_dc = tc.data['Daily Count'].reset_index()
        raise ValueError("don't pass PTCs into this function!")
    else:
        tc_dc = tc.data.reset_index()
    tc_dc['Day of Week'] = tc_dc['Date'].dt.dayofweek

    for _, row in tc_dc.iterrows():
        ryear, rdow = row['Year'], row['Day of Week']

        for p in neighbour_ptcs:

            # Assign temporary variables to make the code below succinct.
            ptc_mse_ydow = p.data['MSE Annual-DoW Averages']
            ptc_mse_y = p.data['MSE Annual Averages']

            # If day of week exists in annual-DoW table...
            if rdow in ptc_mse_ydow.index.levels[0]:
                # Obtain available years for DoW and find closest year to
                # STTC observation.
                closest_year = cmc.get_closest_year(
                    ryear, ptc_mse_ydow.loc[rdow].index.values)
                tc_msedata_row = ptc_mse_ydow.loc[(rdow, closest_year)].values
            else:
                # If not, use annual average.
                closest_year = cmc.get_closest_year(
                    ryear, ptc_mse_y.index.values)
                tc_msedata_row = ptc_mse_y.loc[closest_year].values

            if np.nan in tc_msedata_row:
                raise ValueError("nan discovered during matching {0} with "
                                 "{1}.".format(tc.count_id, p.count_id))

            aadt_closest_year = p.data['AADT'].at[closest_year, 'AADT']

            tc_msedata.append(
                (ryear, row['Day of Year'], p.count_id) +
                tuple(tc_msedata_row) + (closest_year, aadt_closest_year))

    tc_mse = pd.DataFrame(
        tc_msedata,
        columns=('Year', 'Day of Year', 'PTC ID', 'PTC Day-to-AADT Ratio',
                 'PTC MADT Avg.', 'PTC DoM Factor Avg.', 'PTC DoMADT Avg.',
                 'PTC Closest Year', 'PTC Closest Year AADT'))

    if tc_mse.shape[0] != (n_neighbours * tc_dc.shape[0]):
        raise ValueError("missing or added rows in tc_mse for "
                         "{0}.".format(tc.count_id))

    return pd.merge(tc_dc, tc_mse, on=('Year', 'Day of Year'))


def get_aadt_estimate_for_sttc(tc, rdr, citywide_growth_factor, want_year):

    # I disagree with this, but line 95 of DoMSTTC.m seems to do it.
    mean_tc_count = tc.tc_mse['Daily Count'].mean()

    # Get estimates for STTC AADT and MADT.
    tc.tc_mse['AADT_prelim'] = (
        mean_tc_count * tc.tc_mse['PTC Day-to-AADT Ratio'] *
        citywide_growth_factor**(want_year - tc.tc_mse['Year']))
    tc.tc_mse['MADT_pj'] = (
        tc.tc_mse['Daily Count'] * tc.tc_mse['PTC DoM Factor Avg.'] *
        citywide_growth_factor**(want_year - tc.tc_mse['Year']))

    # Determine mean square deviation between normalized monthly patterns.
    tc.tc_mse['MF_STTC'] = tc.tc_mse['MADT_pj'] / tc.tc_mse['AADT_prelim']
    tc.tc_mse['MF_PTC'] = (tc.tc_mse['PTC MADT Avg.'] /
                           tc.tc_mse['PTC Closest Year AADT'])
    tc.tc_mse['Square Deviation'] = (
        tc.tc_mse['MF_STTC'] - tc.tc_mse['MF_PTC'])**2

    # Determine minimum MSE between STTC and each PTC.
    dijs = (tc.tc_mse
            .groupby('PTC ID')[['Square Deviation', 'PTC Day-to-AADT Ratio']]
            .mean())
    ptcid_mmse = dijs['Square Deviation'].idxmin()
    dij_mmseptc = dijs.at[ptcid_mmse, 'PTC Day-to-AADT Ratio']

    # Determine average daily count for most recent year to wanted year.
    closest_year = cmc.get_closest_year(
        want_year, tc.data.index.levels[0].values)
    sttc_daily_count_cyavg = tc.data.loc[closest_year]['Daily Count'].mean()
    aadt_estimate = (sttc_daily_count_cyavg * dij_mmseptc *
                     citywide_growth_factor**(want_year - closest_year))

    return {'Count ID': tc.count_id, 'PTC ID': ptcid_mmse,
            'D_ij': dij_mmseptc, 'Closest Year': closest_year,
            'AADT Estimate': aadt_estimate}


def estimate_aadts(rdr, nb, want_year, n_neighbours=5,
                   single_direction=True,
                   progress_bar=False):

    # Pre-calculate PTC averages.
    for p in tqdm(rdr.ptcs.values(),
                  desc='Calculating PTC averages',
                  disable=(not progress_bar)):
        mse_preprocess_ptc(p)

    # Obtain citywide growth factor.
    citywide_growth_factor = cmc.get_citywide_growth_factor(rdr)

    # Calculate STTC monthly patterns using `n_neighbours` nearest PTCs.
    for tc in tqdm(rdr.sttcs.values(),
                   desc='Calculating STTC monthly patterns',
                   disable=(not progress_bar)):
        tc.tc_mse = get_normalized_seasonal_patterns(
            tc, rdr.ptcs, nb, want_year, n_neighbours=n_neighbours,
            single_direction=single_direction)

    # Process nearest PTC comparisons to estimate AADT for each STTC.
    aadt_estimates = []
    for tc in tqdm(rdr.sttcs.values(),
                   desc='Minimum MSE AADT estimate',
                   disable=(not progress_bar)):
        aadt_estimates.append(
            get_aadt_estimate_for_sttc(tc, rdr,
                                       citywide_growth_factor, want_year))

    return pd.DataFrame(aadt_estimates)[
        ["Count ID", "PTC ID", "Closest Year", "D_ij", "AADT Estimate"]]
