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


def get_sample_request(sttc, year, ptc_year):
    date_sample = pd.DataFrame({
        'Date': sttc.data.loc[year]['Date'].apply(
            lambda x: x.replace(year=ptc_year))})
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


def generate_test_data(rdr):

    # Retrieve and shuffle STTC count IDs.
    sttc_count_ids = np.array(sorted(rdr.sttcs.keys()))
    sttc_negative_count_ids = sttc_count_ids[sttc_count_ids < 0]
    sttc_positive_count_ids = sttc_count_ids[sttc_count_ids > 0]
    np.random.shuffle(sttc_negative_count_ids)
    np.random.shuffle(sttc_positive_count_ids)

    sttcs = []
    for ptc in rdr.ptcs.values():
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

        sttcs.append([ptc.count_id, data_imprinter(ptc, sttc)])

    rdr_dummy = namedtuple('rdr_dummy', ['ptcs', 'sttcs'])
    return rdr_dummy(ptcs=rdr.ptcs, sttcs=dict(sttcs))


def generate_test_database(rdr, n_sets=100, progress_bar=False):
    datasets = []
    for _ in tqdm(range(n_sets),
                  desc='Generating data',
                  disable=(not progress_bar)):
        datasets.append(generate_test_data(rdr))
    return datasets
