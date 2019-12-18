import numpy as np
import pandas as pd
import sys
from tqdm.auto import tqdm

import countmatch_common as cmc
import countmatch_hybrid as cmh


def get_madt_estimate(tc, ptc, want_year, growth_factor):
    ptc_match = cmh.get_ptc_match(tc, ptc)
    ptc_match['MADT_est'] = (
        ptc_match['Daily Count'] * ptc_match['DoM_ijd'] *
        growth_factor**(want_year - ptc_match['Year']))
    madt_est = pd.DataFrame(
        {'MADT_estimate': ptc_match.groupby('Month')['MADT_est'].mean()})
    madt_est['AADT_est'] = (
        ptc_match['Daily Count'] * ptc_match['D_ijd'] *
        growth_factor**(want_year - ptc_match['Year'])).mean()

    return ptc_match, madt_est


def estimate_mse(madt_est, ptc, want_year):
    # Get PTC normalized monthly pattern.
    ptc_closest_year = cmc.get_closest_year(
        want_year, ptc.data['AADT'].index.values)
    # Ensure idempotence of madt_est.
    madt_temp = madt_est.copy()
    madt_temp['MF_STTC'] = madt_temp['MADT_estimate'] / madt_temp['AADT_est']
    madt_temp['MF_PTC'] = (ptc.data['MADT'].loc[ptc_closest_year, 'MADT'] /
                           ptc.data['AADT'].loc[ptc_closest_year, 'AADT'])

    mse = np.mean((madt_temp['MF_STTC'] - madt_temp['MF_PTC'])**2)
    if mse < sys.float_info.epsilon:
        mse = 0.

    return mse


def estimate_cov(madt_est, ptc, want_year):
    ptc_closest_year = cmc.get_closest_year(
        want_year, ptc.data['AADT'].index.values)
    madt_temp = madt_est.copy()
    # No need to growth the PTC MADT, since the growth factor doesn't change
    # month to month.
    madt_temp['MADT_PTC'] = ptc.data['MADT'].loc[ptc_closest_year, 'MADT']
    ratio = madt_temp['MADT_estimate'] / madt_temp['MADT_PTC']
    cov = ratio.std() / ratio.mean()
    return max(0., cov)


def get_aadt_estimate_for_sttc(
        tc, ptcs, nb, want_year, error_estimator, n_neighbours=5,
        single_direction=True, override_growth_factor=False):

    # Find nearest neighbours in same direction.
    neighbour_ptcs = cmc.get_neighbours(
        tc, ptcs, nb, n_neighbours=n_neighbours,
        single_direction=single_direction)

    # Determine best pattern match from nearby PTCs.
    error_metrics = []
    growth_factor_est = (override_growth_factor if override_growth_factor
                         else neighbour_ptcs[0].growth_factor)
    ptc_match, madt_est = get_madt_estimate(
        tc, neighbour_ptcs[0], want_year, growth_factor_est)
    for ptc in neighbour_ptcs:
        error_metrics.append(error_estimator(madt_est, ptc, want_year))
    # Retrieve `ptc_match` table of minimum MSE PTC.
    i_minerr = np.argmin(error_metrics)

    # If we're not just using ptc_match from neighbour_ptcs[0]...
    if i_minerr > 0:
        ptc_match = get_madt_estimate(
            tc, neighbour_ptcs[i_minerr], want_year, growth_factor_est)[0]

    growth_factor = (override_growth_factor if override_growth_factor
                     else neighbour_ptcs[i_minerr].growth_factor)

    # Estimate AADT using most recent year of STTC counts and
    # MADT pattern of closest PTC.
    closest_year = cmc.get_closest_year(
        want_year, ptc_match['Year'].unique())
    mmse_ptc_match_cy = ptc_match.loc[ptc_match['Year'] == closest_year, :]
    aadt_est_closest_year = (
        mmse_ptc_match_cy['Daily Count'] * mmse_ptc_match_cy['D_ijd'] *
        growth_factor**(want_year - mmse_ptc_match_cy['Year'])).mean()

    if np.isnan(aadt_est_closest_year):
        raise ValueError('estimated AADT for {0} is NaN', tc.count_id)

    return (tc.count_id, aadt_est_closest_year)


def estimate_aadts(rdr, nb, want_year, n_neighbours=5,
                   single_direction=True, erroralgo='Minimum MSE',
                   override_growth_factor=False,
                   progress_bar=False):

    # Spot-check which error minimizing algorithm we're using.
    assert erroralgo in ('Minimum MSE', 'Minimum COV'), (
        "erroralgo = {0} unsupported".format(erroralgo))
    error_estimator = (estimate_cov if erroralgo == 'Minimum COV'
                       else estimate_mse)

    # Preprocess PTCs.
    for ptc in tqdm(rdr.ptcs.values(),
                    desc='Preprocessing PTCs',
                    disable=(not progress_bar)):
        cmh.get_Dijd(ptc)
        cmh.get_DoMi(ptc)
        cmh.get_available_years(ptc)
        cmh.get_unstacked_factors(ptc)

    if override_growth_factor:
        citywide_growth_factor = cmc.get_citywide_growth_factor(rdr)

    # Process nearest PTC comparisons to estimate AADT for each STTC.
    aadt_estimates = []
    for tc in tqdm(rdr.sttcs.values(),
                   desc='Estimating STTC AADTs',
                   disable=(not progress_bar)):
        aadt_estimates.append(
            get_aadt_estimate_for_sttc(
                tc, rdr.ptcs, nb, want_year, error_estimator,
                n_neighbours=n_neighbours, single_direction=single_direction,
                override_growth_factor=citywide_growth_factor))

    return pd.DataFrame(aadt_estimates,
                        columns=('Count ID', 'AADT Estimate'))
