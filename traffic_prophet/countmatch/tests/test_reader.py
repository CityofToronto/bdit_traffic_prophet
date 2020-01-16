import pytest
import numpy as np
import pandas as pd

from ...data import SAMPLE_ZIP
from .. import base
from .. import reader


@pytest.fixture(scope="module")
def rdr(cfgcm_test):
    return reader.ReaderZip(SAMPLE_ZIP, cfg=cfgcm_test)


@pytest.fixture(scope="module")
def counts(rdr):
    zr = rdr.get_zipreader(SAMPLE_ZIP['2010'])
    return list(zr)


class TestRawAnnualCount:
    """Test preprocessing routines in RawAnnualCount."""

    def test_from_raw_data(self, rdr):
        zr = rdr.get_zipreader(SAMPLE_ZIP['2010'])
        data = rdr.preprocess_count_data(list(zr)[7])

        sttc_ac = reader.RawAnnualCount.from_raw_data(data)
        assert isinstance(sttc_ac, reader.RawAnnualCount)
        assert sttc_ac.centreline_id == data['centreline_id']
        assert sttc_ac.direction == data['direction']
        assert sttc_ac.year == data['year']
        assert isinstance(sttc_ac.data, pd.DataFrame)


class TestReaderZip:
    """Test ReaderZip for reading in sequences of counts.

    Also test ReaderBase, since that cannot easily be tested standalone.
    """

    def test_initialization(self, cfgcm_test):
        expected_list = sorted(list(SAMPLE_ZIP.values()))

        rdr = reader.ReaderZip(SAMPLE_ZIP, cfg=cfgcm_test)
        assert rdr.source == expected_list

        # Try again but with a string.
        path = "/".join(SAMPLE_ZIP['2010'].split("/")[:-1]) + "/"
        rdr = reader.ReaderZip(path + "15_min*.zip", cfg=cfgcm_test)
        assert rdr.source == expected_list

        # Try again but with a list.
        sample = [path + "15_min_counts_2011_neg_sample.zip",
                  path + "15_min_counts_2011_pos_sample.zip",
                  path + "15_min_counts_2012_neg_sample.zip",
                  path + "15_min_counts_2010_neg_sample.zip"]
        rdr = reader.ReaderZip(sample, cfg=cfgcm_test)
        assert rdr.source == expected_list

    def test_get_zipreader(self, counts):
        # counts was already generated at setup; this checks its output.

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

    def test_regularize_timeseries(self, rdr, counts):
        # counts[7] is -890 for 2010 (holdover from an earlier generation of
        # test suite).
        ref_data = counts[7]

        crd = rdr.regularize_timeseries(ref_data)
        assert (sorted(crd.columns) ==
                sorted(['Timestamp', 'Count']))
        assert crd.shape == (ref_data['data'].shape[0], 2)
        assert np.array_equal(crd['Timestamp'].dt.minute.unique(),
                              np.array([0, 15, 30, 45]))
        assert np.all(crd['Count'] ==
                      ref_data['data']['Count'].astype(float))

        # Introduce some errors to counts[2] (also a holdover).
        fake_data = counts[2].copy()
        # Do a deep copy just in case.
        fake_data['data'] = fake_data['data'].copy()
        fake_data['data'].at[30, 'Timestamp'] = (
            pd.Timestamp('2010-06-09 07:39:31'))
        fake_data['data'].at[30, 'Count'] = 235
        fake_data['data'].at[32, 'Timestamp'] = (
            pd.Timestamp('2010-06-09 07:51:12'))
        crd2 = rdr.regularize_timeseries(fake_data)
        assert crd2.shape == (counts[2]['data'].shape[0] - 2, 2)
        assert np.array_equal(crd2['Timestamp'].dt.minute.unique(),
                              np.array([0, 15, 30, 45]))
        assert crd2.at[30, 'Timestamp'] == pd.Timestamp('2010-06-09 07:45:00')
        assert np.isclose(crd2.at[30, 'Count'], 93., rtol=1e-10)

    def test_preprocess_count_data(self, rdr, counts):
        ref = counts[7]
        rd = ref.copy()
        # Do a deep copy because preprocess_count_data alters its arguments.
        rd['data'] = rd['data'].copy()

        rd = rdr.preprocess_count_data(rd)

        # Ensure every unique date is represented.
        assert rd['data'].shape == (
            ref['data']['Timestamp'].dt.date.unique().shape[0], 2)
        assert not np.any(rd['data'].index.duplicated())

        # Check that index values represent days of year.
        assert np.array_equal(rd['data'].index.values,
                              rd['data']['Date'].dt.dayofyear)

        # Fuzz test to see if we've summed the counts up properly.
        for cidx in np.random.choice(rd['data'].index.values, size=10):
            cdate = rd['data'].at[cidx, 'Date'].date()
            assert np.isclose(
                rd['data'].at[cidx, 'Daily Count'],
                ref['data'].loc[(ref['data']['Timestamp'].dt.date ==
                                 cdate), 'Count'].sum(),
                rtol=1e-10)

        # Check that days with too few counts don't end up in daily counts.
        # crd_partial has the first three full days of the year, with 73 counts
        # on the second day removed.
        rdd_partial = (ref['data'].iloc[:288, :]
                       .drop(index=range(108, 181))
                       .reset_index(drop=True))
        assert np.array_equal(
            rdd_partial['Timestamp'].dt.dayofyear.unique(),
            np.array([1, 2, 3]))
        rdp = ref.copy()
        rdp['data'] = rdd_partial
        rdp = rdr.preprocess_count_data(rdp)
        assert np.array_equal(rdp['data']['Date'].dt.dayofyear,
                              np.array([1, 3]))

    def test_append_counts(self, cfgcm_test):
        # Appending counts alters arguments, so can't rely on fixture here.
        rdr = reader.ReaderZip(SAMPLE_ZIP, cfg=cfgcm_test)
        counts_2010 = [
            reader.RawAnnualCount.from_raw_data(
                rdr.preprocess_count_data(c))
            for c in rdr.get_zipreader(SAMPLE_ZIP['2010'])]
        counts_2012 = [
            reader.RawAnnualCount.from_raw_data(
                rdr.preprocess_count_data(c))
            for c in rdr.get_zipreader(SAMPLE_ZIP['2012'])]
        counts = {}
        rdr.append_counts(counts_2010, counts)
        rdr.append_counts(counts_2012, counts)

        # Annoyingly, there is no overlap in PTC locations between 2010
        # and 2011.
        assert (sorted(counts.keys()) ==
                [-446378, -104870, -1978, -890, -487, -427, -410, -252, -241])

        # Check that counts from multiple years are stored under the same key.
        assert len(counts[-104870]) == 2
        assert len(counts[-241]) == 2

        # Check table integrity.
        assert counts_2010[0].data.equals(counts[-104870][0].data)
        assert counts_2012[4].data.equals(counts[-890][1].data)

    def test_unify_counts(self, cfgcm_test):
        rdr = reader.ReaderZip(SAMPLE_ZIP, cfg=cfgcm_test)
        counts_2010 = [
            reader.RawAnnualCount.from_raw_data(
                rdr.preprocess_count_data(c))
            for c in rdr.get_zipreader(SAMPLE_ZIP['2010'])]
        counts_2011p = [
            reader.RawAnnualCount.from_raw_data(
                rdr.preprocess_count_data(c))
            for c in rdr.get_zipreader(SAMPLE_ZIP['2011p'])]
        counts_2012 = [
            reader.RawAnnualCount.from_raw_data(
                rdr.preprocess_count_data(c))
            for c in rdr.get_zipreader(SAMPLE_ZIP['2012'])]

        counts = {}
        rdr.append_counts(counts_2010, counts)
        rdr.append_counts(counts_2011p, counts)
        rdr.append_counts(counts_2012, counts)

        rdr.unify_counts(counts)

        assert (sorted(counts.keys()) ==
                [-446378, -104870, -1978, -890, -487, -427, -410, -252, -241,
                 170, 104870])
        # Check that we've concatenated three years' data together.
        # `unify_counts` alters `counts`.  Since those do not make copies of
        # data from `counts_2010` and `counts_2012`, those are altered as well.
        # This only matters for testing, however.
        assert counts[-104870].count_id == -104870
        assert counts[-104870].centreline_id == 104870
        assert counts[-104870].direction == -1
        # No count should be labeled permanent yet.
        assert not counts[-104870].is_permanent
        assert counts[-104870].data.equals(
            pd.concat([counts_2010[0].data, counts_2012[0].data]))

        assert counts[-241].count_id == -241
        assert counts[-241].centreline_id == 241
        assert counts[-241].direction == -1
        assert not counts[-241].is_permanent
        assert counts[-241].data.equals(
            pd.concat([counts_2010[1].data, counts_2012[2].data]))

        assert counts[170].count_id == 170
        assert counts[170].centreline_id == 170
        assert counts[170].direction == 1
        assert not counts[170].is_permanent
        assert counts[170].data.equals(counts_2011p[0].data)

    def test_read(self, cfgcm_test):
        rdr = reader.ReaderZip(SAMPLE_ZIP, cfg=cfgcm_test)
        rdr.read()

        assert (sorted(rdr.counts.keys()) ==
                [-446378, -104870, -1978, -890, -680, -487, -427, -410,
                 -252, -241, -170, 170, 104870])
        assert isinstance(rdr.counts[-241], base.Count)

        # Check that all available years have been read in.
        included_years = []
        for key in rdr.counts.keys():
            included_years += list(rdr.counts[key].data.index.levels[0].values)
        assert sorted(list(set(included_years))) == [2010, 2011, 2012]

        # Check that reading a non-zip raises an error.
        with pytest.raises(IOError):
            path = "/".join(SAMPLE_ZIP['2010'].split("/")[:-1]) + "/"
            rdr = reader.ReaderZip(path + 'count_lonlat.csv')
            rdr.read()


class TestReaderPostgres:
    """Test ReaderPostgres for reading in sequences of counts.

    Currently only `preprocess_count_data` can be tested, since everything else
    requires a test Postgres DB to be set up during testing.
    """

    def test_initialization(self):
        rdr = reader.ReaderPostgres('Placeholder')
        assert rdr.counts is None
        assert rdr.source == 'Placeholder'

    def test_preprocess_count_data(self, counts, cfgcm_test):
        rdr = reader.ReaderPostgres(SAMPLE_ZIP, cfg=cfgcm_test)
        # Emulate daily count raw data from Postgres.
        rd = counts[7].copy()
        rd['data'] = rd['data'].copy()
        rd['data']['Date'] = rd['data']['Timestamp'].dt.date
        crdg = rd['data'].groupby('Date')
        daily_count = pd.DataFrame({
            'Daily Count': 96. * crdg['Count'].mean()})
        daily_count.reset_index(inplace=True)
        daily_count['Date'] = pd.to_datetime(daily_count['Date'])
        rd['data'] = daily_count

        # Preprocess count data.
        rd = rdr.preprocess_count_data(rd)

        # Ensure every unique date is represented.
        assert not np.any(rd['data'].index.duplicated())

        # Check that index values represent days of year.
        assert np.array_equal(rd['data'].index.values,
                              rd['data']['Date'].dt.dayofyear)


class TestReaderFunction:
    """Test reader function for returning the right class."""

    def test_reader(self, cfgcm_test):
        rdr = reader.read(SAMPLE_ZIP['2011p'], cfg=cfgcm_test)
        assert sorted(rdr.counts.keys()) == [170, 104870]
        assert rdr.counts[104870].data.shape == (3, 2)
        assert (rdr.counts[104870].data.index.levels[0].values ==
                np.array([2011]))
