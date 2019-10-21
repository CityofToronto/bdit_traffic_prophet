import pytest
import numpy as np
import pandas as pd

from ...data import SAMPLE_ZIP
from .. import reader


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

# class TestAnnualDailyCount:

#     @pytest.mark.parametrize('filenumber', list(range())
#     def test_round_timestamp(self, filenumber):
#         dc = cm.AnnualDailyCount(0, 0, None)
#         # Ensure randomly drawn data has same resolution as actual input data.
#         t_pd = pd.to_datetime(t).round('1s')

#         assert abs((t_rounded - t_pd) / np.timedelta64(1, 'm')) <= 7.5
