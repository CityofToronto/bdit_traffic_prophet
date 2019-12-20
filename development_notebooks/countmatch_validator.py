import numpy as np
import pandas as pd
from collections import namedtuple
from tqdm.auto import tqdm

import countmatch_common as cmc

import sys
sys.path.append('../')
from traffic_prophet.countmatch import reader


def get_year_mapping(sttc_years, ptc_years):
    # Produce a one-to-one mapping between STTC and PTC year.
    if len(sttc_years) > len(ptc_years):
        raise ValueError('more STTC years required than available in PTC!')

    if len(sttc_years) == 1:
        closest_year = cmc.get_closest_year(sttc_years[0], np.array(ptc_years))
        return {sttc_years[0]: closest_year}

    # Create sorted copies of the years.
    sttc_years = np.sort(sttc_years)
    ptc_years = np.sort(ptc_years)

    # Create 'rescaled_sttc_years', so that get_closest_year can map two
    # very different year ranges together.  I'd prefer to do a fit here,
    # but we can't guarantee sttc_years has the same length as ptc_years.
    slope = ((ptc_years[-1] - ptc_years[0]) /
             (sttc_years[-1] - sttc_years[0]))
    rescaled_sttc_years = slope * (sttc_years - sttc_years[0]) + ptc_years[0]
    closest_years = cmc.get_closest_year(rescaled_sttc_years, ptc_years)

    year_mapping = {}
    for i in range(len(sttc_years)):
        if closest_years[i] not in year_mapping.values():
            year_mapping[sttc_years[i]] = closest_years[i]

    return year_mapping


def shift_year(d, ptc_year):
    try:
        return d.replace(year=ptc_year)
    except ValueError:
        return d.replace(year=ptc_year, day=(d.day - 1))


def get_sample_request(sttc, year, ptc_year):
    date_sample = pd.DataFrame({
        'Date': sttc.data.loc[year]['Date'].apply(
            shift_year, args=(ptc_year,))})
    date_sample['Month'] = date_sample['Date'].dt.month
    date_sample['Day of Week'] = date_sample['Date'].dt.month
    return (date_sample.groupby('Month')['Date']
            .agg(['min', 'count'])
            .rename(columns={'min': 'Start Date', 'count': 'N_days'}))


def nearest_idx_factory(ptc_daily_counts):
    ptc_date_idxs = pd.Series(ptc_daily_counts.index,
                              index=ptc_daily_counts['Date'].values)
    return lambda x: ptc_date_idxs.index.get_loc(x, method='nearest')


def data_imprinter(ptc, sttc):
    sttc_years = sttc.data['Date'].index.levels[0].values
    ptc_years = ptc.data['AADT'].index.values
    year_mapping = get_year_mapping(sttc_years, ptc_years)

    daily_counts = []
    for year in year_mapping.keys():
        ptc_year = year_mapping[year]
        sample_request = get_sample_request(sttc, year, ptc_year)

        ptc_daily_counts = ptc.data['Daily Count'].loc[ptc_year]
        nearest_idx = nearest_idx_factory(ptc_daily_counts)

        iloc_idxs = []
        for _, row in sample_request.iterrows():
            idx = nearest_idx(row['Start Date'])
            iloc_idxs += list(range(idx, idx + row['N_days']))

        # Prune indices that go beyond bounds of `ptc_daily_counts`.
        iloc_idxs = np.array(iloc_idxs)
        iloc_idxs = iloc_idxs[iloc_idxs < ptc_daily_counts.shape[0]]

        sample = ptc_daily_counts.iloc[iloc_idxs, :].copy()
        sample.index = pd.MultiIndex.from_product(
            [[ptc_year, ], sample.index],
            names=['Year', sample.index.name])
        daily_counts.append(sample)

    return reader.Count(
        ptc.count_id, ptc.centreline_id, ptc.direction,
        pd.concat(daily_counts), is_permanent=False)


def generate_test_data(rdr, restrict_to_year=False):

    # Retrieve and shuffle STTC count IDs.
    sttc_count_ids = np.array(sorted(rdr.sttcs.keys()))
    sttc_negative_count_ids = sttc_count_ids[sttc_count_ids < 0]
    sttc_positive_count_ids = sttc_count_ids[sttc_count_ids > 0]
    np.random.shuffle(sttc_negative_count_ids)
    np.random.shuffle(sttc_positive_count_ids)

    # Use a subset of PTCs that have the year we're looking for available, or
    # just use all available PTCs.
    ptcs = ([x for x in rdr.ptcs.values()
             if restrict_to_year in x.data['AADT'].index] if restrict_to_year
            else rdr.ptcs.values())

    if not len(ptcs):
        raise ValueError("no PTCs available with year {0}!"
                         .format(restrict_to_year))

    test_sttcs = []
    for ptc in ptcs:
        sttc_count_ids = (sttc_negative_count_ids
                          if ptc.direction < 0 else sttc_positive_count_ids)
        # Select an STTC.
        for i in range(len(sttc_count_ids)):
            sttc = rdr.sttcs[sttc_count_ids[i]]
            sttc_years = sttc.data['Date'].index.levels[0].values
            ptc_years = ptc.data['AADT'].index.values
            # Check if the STTC has more years of data than the PTC - if not,
            # use the chosen STTC.
            if sttc_years.shape[0] <= ptc_years.shape[0]:
                break

        if i == range(len(sttc_count_ids)):
            raise ValueError('ran out of STTCs!')

        # Remove the STTC ID from future consideration.
        sttc_count_ids = np.delete(sttc_count_ids, i)

        test_sttcs.append([ptc.count_id, data_imprinter(ptc, sttc)])

    rdr_dummy = namedtuple('rdr_dummy', ['ptcs', 'sttcs'])
    return rdr_dummy(ptcs=rdr.ptcs, sttcs=dict(test_sttcs))


def generate_test_database(rdr, n_sets=100, restrict_to_year=False,
                           progress_bar=False):
    datasets = []
    for _ in tqdm(range(n_sets),
                  desc='Generating data',
                  disable=(not progress_bar)):
        datasets.append(
            generate_test_data(rdr, restrict_to_year=restrict_to_year))
    return datasets


def run_algorithm_on_test_data(dataset, algo, algo_args, nb, want_year,
                               progress_bar=False):

    raw_estimates = []
    for dset in tqdm(dataset, desc=("Processing {0}".format(want_year)),
                     disable=(not progress_bar)):
        raw_estimates.append(
            algo(dset, nb, want_year, **algo_args))

    # Get AADT estimates for each run in dataset.
    aadts = pd.concat(
        [(x.sort_values('Count ID')[['Count ID', 'AADT Estimate']]
          .set_index('Count ID', drop=True))
         for x in raw_estimates], axis=1)
    aadts.columns = list(range(len(raw_estimates)))

    # Match AADT ground truths to count IDs.
    aadts_gt = [dset.ptcs[x].data['AADT'].at[want_year, 'AADT']
                for x in aadts.index.values]
    aadts['Ground Truth'] = aadts_gt

    abs_errors = np.abs(aadts[list(range(len(raw_estimates)))].values -
                        aadts[['Ground Truth']].values)

    aadts['MAE'] = abs_errors.mean(axis=1)
    aadts['STDAE'] = abs_errors.std(axis=1)

    aadts['Year'] = want_year

    return aadts


def validation(rdr, nb, algo, algo_args={},
               n_sets=100, progress_bar=False):

    # Obtain the set of all years where we have PTCs.
    all_available_PTC_years = np.unique(np.concatenate(
        [x.data['AADT'].index.values for x in rdr.ptcs.values()]))

    datasets = []
    for year in all_available_PTC_years:
        if progress_bar:
            print("For year {0}".format(year))
        datasets.append((
            year,
            generate_test_database(rdr, n_sets=n_sets,
                                   restrict_to_year=year,
                                   progress_bar=progress_bar)))

    aadt_validation = []
    for want_year, dataset in datasets:
        aadt_validation.append(
            run_algorithm_on_test_data(
                dataset, algo, algo_args, nb, want_year,
                progress_bar=progress_bar))

    return datasets, aadt_validation
