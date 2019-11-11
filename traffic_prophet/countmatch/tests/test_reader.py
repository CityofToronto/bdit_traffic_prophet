import pytest
import numpy as np
import pandas as pd

from ...data import SAMPLE_ZIP
from .. import reader
from ...connection import Connection


class TestAnnualCount:
    """Test preprocessing routines in AnnualCount."""

    def setup(self):
        rdr = reader.Reader(SAMPLE_ZIP)
        zr = rdr.get_zipreader(SAMPLE_ZIP['2010'])
        self.counts = list(zr)
        self.sttc_data = self.counts[2]
        self.ptc_data = self.counts[7]

    def test_regularize_timeseries(self):
        # Test processing a file that has no rounding issues.
        ac = reader.AnnualCount(1000, -1, 2010, None)
        crd = ac.regularize_timeseries(self.ptc_data)
        assert (sorted(crd.columns) ==
                sorted(['Timestamp', 'Date', 'Month', 'Day of Week', 'Count']))
        assert crd.shape == (self.ptc_data['data'].shape[0], 5)
        assert np.array_equal(crd['Timestamp'].dt.minute.unique(),
                              np.array([0, 15, 30, 45]))
        assert np.all(crd['Count'] ==
                      self.ptc_data['data']['Count'].astype(float))
        assert np.all(crd['Day of Week'] ==
                      self.ptc_data['data']['Timestamp'].dt.dayofweek)

        # Introduce some errors to sttc_data.
        fake_sttc_data = self.sttc_data.copy()
        # Do a deep copy just in case.
        fake_sttc_data['data'] = self.sttc_data['data'].copy()
        fake_sttc_data['data'].at[30, 'Timestamp'] = (
            pd.Timestamp('2010-06-09 07:39:31'))
        fake_sttc_data['data'].at[30, 'Count'] = 235
        fake_sttc_data['data'].at[32, 'Timestamp'] = (
            pd.Timestamp('2010-06-09 07:51:12'))
        crd2 = ac.regularize_timeseries(fake_sttc_data)
        assert crd2.shape == (self.sttc_data['data'].shape[0] - 2, 5)
        assert np.array_equal(crd2['Timestamp'].dt.minute.unique(),
                              np.array([0, 15, 30, 45]))
        assert crd2.at[30, 'Timestamp'] == pd.Timestamp('2010-06-09 07:45:00')
        assert np.isclose(crd2.at[30, 'Count'], 93., rtol=1e-10)

    def test_process_15min_count_data(self):
        ac = reader.AnnualCount(1000, -1, 2010, None)
        crd = ac.regularize_timeseries(self.ptc_data)
        daily_count = ac.process_15min_count_data(crd)
        ac.reset_daily_count_index(daily_count)

        # Ensure every unique date is represented.
        assert daily_count.shape == (crd['Date'].unique().shape[0], 2)
        assert not np.any(daily_count.index.duplicated())

        # Check that index values represent days of year.
        assert np.array_equal(daily_count.index.values,
                              daily_count['Date'].dt.dayofyear)

        # Fuzz test to see if we've summed the counts up properly.
        for cidx in np.random.choice(daily_count.index.values, size=10):
            cdate = daily_count.at[cidx, 'Date'].date()
            assert np.isclose(
                daily_count.at[cidx, 'Daily Count'],
                crd.loc[crd['Date'] == cdate, 'Count'].sum(),
                rtol=1e-10)

    def test_is_permanent(self):
        known_ptc_ids = [890, 104870]
        ac = reader.AnnualCount(1000, -1, 2010, None)
        for c in self.counts:
            if c['centreline_id'] in known_ptc_ids:
                assert ac.is_permanent_count(self.ptc_data)
            else:
                assert not ac.is_permanent_count(self.sttc_data)

    def test_process_permanent_count_data(self):
        ac = reader.AnnualCount(1000, -1, 2010, None)
        # We'll use only one day from December to check what the algorithm will
        # do in the case of sparse data.
        temp_data = self.ptc_data.copy()
        temp_data['data'] = self.ptc_data['data'].loc[:24863, :].copy()
        crd = ac.regularize_timeseries(temp_data)
        po = ac.process_permanent_count_data(crd)

        po_m = po['MADT']
        assert ((po_m['MADT'] *
                 crd.groupby('Month')['Count'].count() / 96.).sum() ==
                crd['Count'].sum())
        # 2010 is not a leap year.
        assert po_m['Days in Month'].sum() == 365

        # Every month and day of week must be included in domadt.
        assert po['DoMADT'].shape == (12, 7)
        n_dom = (crd.groupby(['Month', 'Day of Week'])['Count']
                 .count().unstack() // 96)
        sum_dom = (crd.groupby(['Month', 'Day of Week'])['Count']
                   .sum().unstack())
        assert np.allclose(po['DoMADT'] * n_dom, sum_dom,
                           rtol=1e-10, equal_nan=True)

        assert po['DoM Factor'].shape == (12, 7)
        po_factortimesdom = (po['DoM Factor'] * po['DoMADT']).values
        repeated_madt = np.repeat(po['MADT']['MADT'].values[:, np.newaxis],
                                  7, axis=1)
        assert np.allclose(po_factortimesdom[:11, :],
                           repeated_madt[:11, :], rtol=1e-10)
        dec1_dow = crd.iat[-1, 0].dayofweek
        assert np.isclose(po_factortimesdom[11, dec1_dow],
                          repeated_madt[11, dec1_dow])

        # Explicitly check that days with no data are NaN.  First, get all
        # days of the week other than the one for Dec # 1.
        weekdays_not_dec1 = list(set(range(7)) - set((dec1_dow, )))
        assert np.all(np.isnan(po['DoMADT'].iloc[-1, weekdays_not_dec1]))
        assert np.all(np.isnan(po['DoM Factor'].iloc[-1, weekdays_not_dec1]))

        aadt = ((po['MADT']['MADT'] *
                 po['MADT']['Days in Month']).sum() /
                po['MADT']['Days in Month'].sum())
        assert np.isclose(po['AADT'], aadt, rtol=1e-10)

    def test_from_raw_data(self):
        sttc_ac = reader.AnnualCount.from_raw_data(self.sttc_data)
        assert isinstance(sttc_ac, reader.AnnualCount)
        assert sttc_ac.centreline_id == self.sttc_data['centreline_id']
        assert sttc_ac.direction == self.sttc_data['direction']
        assert sttc_ac.year == self.sttc_data['year']
        assert isinstance(sttc_ac.data, pd.DataFrame)

        ptc_ac = reader.AnnualCount.from_raw_data(self.ptc_data)
        assert isinstance(ptc_ac.data, dict)
        assert (sorted(ptc_ac.data.keys()) ==
                sorted(['Daily Count', 'MADT', 'DoMADT',
                        'DoM Factor', 'AADT']))


class TestReader:
    """Test Reader for reading in sequences of counts."""

    def test_initialization(self):
        expected_list = sorted(list(SAMPLE_ZIP.values()))

        rdr = reader.Reader(SAMPLE_ZIP)
        assert rdr.sourcetype == 'Zip'
        assert rdr._reader == rdr.read_zip
        assert rdr.source == expected_list

        # Try again but with a string.
        path = "/".join(SAMPLE_ZIP['2010'].split("/")[:-1]) + "/"
        rdr = reader.Reader(path + "15_min*.zip")
        assert rdr._reader == rdr.read_zip
        assert rdr.source == expected_list

        # Try again but with a list.
        sample = [path + "15_min_counts_2011_neg_sample.zip",
                  path + "15_min_counts_2012_neg_sample.zip",
                  path + "15_min_counts_2010_neg_sample.zip"]
        rdr = reader.Reader(sample)
        assert rdr._reader == rdr.read_zip
        assert rdr.source == expected_list

    def test_get_zipreader(self):
        rdr = reader.Reader(SAMPLE_ZIP)
        zr = rdr.get_zipreader(SAMPLE_ZIP['2010'])
        counts = list(zr)

        assert len(counts) == 8
        assert ([c['filename'] for c in counts] ==
                ['104870_11510_2010.txt',
                 '241_4372_2010.txt',
                 '252_4505_2010.txt',
                 '410_8108_2010.txt',
                 '427_8256_2010.txt',
                 '446378_2398_2010.txt',
                 '487_9229_2010.txt',
                 '890_17700_2010.txt'])
        assert ([c['centreline_id'] for c in counts] ==
                [104870, 241, 252, 410, 427, 446378, 487, 890])
        assert ([c['direction'] for c in counts] ==
                [-1 for i in range(len(counts))])
        assert ([c['data'].shape for c in counts] ==
                [(30912, 2), (288, 2), (96, 2), (96, 2), (96, 2), (10752, 2),
                 (288, 2), (27072, 2)])
        assert np.array_equal(
            counts[2]['data'].dtypes.values,
            np.array([np.dtype('<M8[ns]'), np.dtype('int64')]))
        assert (counts[2]['data'].at[3, 'Timestamp'] ==
                pd.to_datetime('2010-06-09 00:45:00'))
        assert counts[2]['data'].at[10, 'Count'] == 1

        # TO DO: once we get a logger going, should probably also check using a
        # tmpdir that we can ignore and log counts too small to be used.

    def test_append_counts(self):
        rdr = reader.Reader(SAMPLE_ZIP)
        counts_2010 = [reader.AnnualCount.from_raw_data(c)
                       for c in rdr.get_zipreader(SAMPLE_ZIP['2010'])]
        counts_2012 = [reader.AnnualCount.from_raw_data(c)
                       for c in rdr.get_zipreader(SAMPLE_ZIP['2012'])]
        ptcs = {}
        sttcs = {}
        rdr.append_counts(counts_2010, ptcs, sttcs)
        rdr.append_counts(counts_2012, ptcs, sttcs)

        # Annoyingly, there is no overlap in PTC locations between 2010
        # and 2011.
        assert sorted(ptcs.keys()) == [890, 104870]
        assert (sorted(sttcs.keys()) ==
                [241, 252, 410, 427, 487, 890, 1978, 446378])

        # Check that counts from multiple years are stored under the same key.
        assert len(ptcs[104870]) == 2
        assert len(sttcs[241]) == 2

        # Check table integrity.
        assert (counts_2010[0].data['DoMADT']
                .equals(ptcs[104870][0].data['DoMADT']))
        assert (counts_2012[4].data['Daily Count']
                .equals(sttcs[890][0].data['Daily Count']))

    def test_check_processed_count_integrity(self):
        rdr = reader.Reader(SAMPLE_ZIP)
        with pytest.raises(ValueError) as exc:
            rdr.check_processed_count_integrity(ptcs={}, sttcs={})
        assert "no count file has more than" in str(exc.value)

    def test_unify_counts(self):
        rdr = reader.Reader(SAMPLE_ZIP)
        counts_2010 = [reader.AnnualCount.from_raw_data(c)
                       for c in rdr.get_zipreader(SAMPLE_ZIP['2010'])]
        counts_2012 = [reader.AnnualCount.from_raw_data(c)
                       for c in rdr.get_zipreader(SAMPLE_ZIP['2012'])]
        ptcs = {}
        sttcs = {}
        rdr.append_counts(counts_2010, ptcs, sttcs)
        rdr.append_counts(counts_2012, ptcs, sttcs)

        rdr.unify_counts(ptcs, sttcs)

        assert sorted(ptcs.keys()) == [890, 104870]
        assert (sorted(sttcs.keys()) ==
                [241, 252, 410, 427, 487, 890, 1978, 446378])
        # Check that we've concatenated two years' data together.
        # `unify_counts` is not idempotent and alters `ptcs` and `sttcs`.
        # Since those do not make copies of data from `counts_2010` and
        # `counts_2012`, those are altered as well.  This only matters for
        # testing, however.
        assert ptcs[104870].centreline_id == 104870
        assert ptcs[104870].direction == -1
        assert ptcs[104870].is_permanent
        assert ptcs[104870].data['DoM Factor'].equals(
            pd.concat([counts_2010[0].data['DoM Factor'],
                       counts_2012[0].data['DoM Factor']]))
        assert sttcs[241].centreline_id == 241
        assert sttcs[241].direction == -1
        assert not sttcs[241].is_permanent
        assert sttcs[241].data.equals(
            pd.concat([counts_2010[1].data, counts_2012[2].data]))

    def test_read_zip(self):
        rdr = reader.Reader(SAMPLE_ZIP)
        rdr.read()

        assert sorted(rdr.ptcs.keys()) == [890, 104870]
        assert (sorted(rdr.sttcs.keys()) ==
                [170, 241, 252, 410, 427, 487, 680, 890, 1978,
                 104870, 446378])
        assert isinstance(rdr.ptcs[104870], reader.Count)
        assert isinstance(rdr.sttcs[241], reader.Count)

        # Check that all available years have been read in.
        included_years = []
        for key in rdr.sttcs.keys():
            included_years += list(rdr.sttcs[key].data.index.levels[0].values)
        assert sorted(list(set(included_years))) == [2010, 2011, 2012]

        # Check that reading a non-zip raises an error.
        with pytest.raises(IOError):
            path = "/".join(SAMPLE_ZIP['2010'].split("/")[:-1]) + "/"
            rdr = reader.Reader(path + 'count_lonlat.csv')
            rdr.read_zip()
