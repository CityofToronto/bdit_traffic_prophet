import pytest
import numpy as np
import pandas as pd

from ...data import SAMPLE_ZIP
from .. import reader


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
        for key in ['Timestamp', 'Date', 'Month', 'Day of Week', 'Count']:
            assert key in crd.columns
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

    def test_process_count_data(self):
        ac = reader.AnnualCount(1000, -1, 2010, None)
        crd = ac.regularize_timeseries(self.ptc_data)
        daily_counts = ac.process_count_data(crd)

        # Ensure every unique date is represented.
        assert daily_counts.shape == (crd['Date'].unique().shape[0], 1)
        assert not np.any(daily_counts.index.duplicated())

        # Fuzz test to see if we've summed the counts up properly.
        for ctime in np.random.choice(daily_counts.index.values, size=10):
            assert np.isclose(
                daily_counts.at[ctime, 'Daily Count'],
                crd.loc[crd['Date'] == ctime, 'Count'].sum(),
                rtol=1e-10)

    def test_is_permanent(self):
        known_ptc_ids = [890, 104870]
        ac = reader.AnnualCount(1000, -1, 2010, None)
        for c in self.counts:
            if c['centreline_id'] in known_ptc_ids:
                assert ac.is_permanent_count(self.ptc_data)
            else:
                assert not ac.is_permanent_count(self.sttc_data)


class TestReader:
    """Test reader.Reader methods."""

    def test_initialization(self):
        """Test object initialization"""

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
        """Test zip reader iterator."""

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

    def test_read_zip(self):
        """Test zip reader."""

        # Check that reading a non-zip raises an error.
        with pytest.raises(IOError):
            path = "/".join(SAMPLE_ZIP['2010'].split("/")[:-1]) + "/"
            rdr = reader.Reader(path + 'count_lonlat.csv')
            counts = rdr.read_zip()
