"""I/O and pre-processing classes for countmatch."""

import numpy as np
import pandas as pd
import zipfile
import glob
import warnings

from .. import cfg
from .. import conn


class Count:

    def __init__(self, count_id, centreline_id, direction,
                 data, is_permanent=False):
        self.count_id = count_id
        self.centreline_id = int(centreline_id)
        self.direction = int(direction)
        self.is_permanent = bool(is_permanent)
        self.data = data


class RawAnnualCount(Count):
    """Storage for total and average daily traffic."""

    def __init__(self, count_id, centreline_id, direction, year,
                 data, is_permanent=False):
        self.year = int(year)
        super().__init__(count_id, centreline_id, direction, data,
                         is_permanent=is_permanent)

    @classmethod
    def from_raw_data(cls, rd):
        """Processes input data.

        Parameters
        ----------
        rd : dict
            Raw data, with entries for 'centreline_id', 'direction',
            'year' and 'data'.  The first three are integers, while 'data'
            is a `pandas.DataFrame` with 'Count' column and either 'Timestamp'
            or 'Date' column.
        """
        if 'Date' not in rd['data'].columns:
            raise ValueError("raw input data missing 'Date' column.")

        # Save to a new object.
        return cls(rd['centreline_id'] * rd['direction'], rd['centreline_id'],
                   rd['direction'], rd['year'], rd['data'])


class ReaderBase:

    def __init__(self, source):
        # Store permanent and temporary stations.
        self.counts = None
        self.source = source

    def read(self):
        """Read source data into a dictionary of counts."""
        # Holds processed counts.
        counts = {}
        # Read counts from source.
        self.read_source(counts)
        # Combine counts at the same location and different years.
        self.unify_counts(counts)
        self.counts = counts

    def read_source(self, counts):
        raise NotImplementedError

    @staticmethod
    def reset_daily_count_index(daily_count):
        """In-place reset of `daily_count` index to be the day of year."""
        daily_count.set_index(
            pd.Index(daily_count['Date'].dt.dayofyear, name='Day of Year'),
            inplace=True)

    @staticmethod
    def has_enough_data(data):
        """Checks if there is enough data to be a usable count."""
        return data.shape[0] >= cfg.cm['min_stn_count']

    @staticmethod
    def append_counts(current_counts, counts):
        for c in current_counts:
            if c.count_id in counts.keys():
                counts[c.count_id].append(c)
            else:
                counts[c.count_id] = [c, ]

    @staticmethod
    def unify_counts(counts):
        """Unify count objects across years.

        Works **in place**, and replaces the index of each table with a
        MultiIndex.
        """
        for cid in counts.keys():
            for item in counts[cid]:
                _ctable = item.data
                _ctable.index = pd.MultiIndex.from_product(
                    [[item.year, ], _ctable.index],
                    names=['Year', _ctable.index.name])
            unified_data = pd.concat([c.data for c in counts[cid]])
            counts[cid] = Count(counts[cid][0].count_id,
                                counts[cid][0].centreline_id,
                                counts[cid][0].direction,
                                unified_data, is_permanent=False)


class ReaderZip(ReaderBase):

    def __init__(self, source):
        if type(source) == str:
            source = glob.glob(source)
        elif type(source) == dict:
            source = [source[k] for k in sorted(source.keys())]
        source = sorted(source)

        super().__init__(source)

    def read_source(self, counts):
        """Read zip file contents into RawAnnualCount objects."""
        # Cycle through all zip files.
        for zf in self.source:
            # Check if the file's actually a zip.
            if not zipfile.is_zipfile(zf):
                raise IOError('{0} is not a zip file.'.format(zf))

            current_counts = [
                RawAnnualCount.from_raw_data(
                    self.preprocess_count_data(c))
                for c in self.get_zipreader(zf)]

            # Append counts to processed count dicts.
            self.append_counts(current_counts, counts)

    def preprocess_count_data(self, rd):
        """Calculates total daily traffic from raw count data."""
        # Copy file and round timestamps to the nearest 15 minutes.
        crd = self.regularize_timeseries(rd)
        # Get daily total count values.
        crdg = crd.groupby('Date')
        # Get the number of bins per day and drop days with fewer than the
        # minimum allowed number of counts.
        valids = crdg['Count'].count() >= cfg.cm['min_counts_in_day']
        # Calculate daily counts.
        daily_count = pd.DataFrame({
            'Daily Count': 96. * crdg['Count'].mean()[valids]})
        daily_count.reset_index(inplace=True)
        daily_count['Date'] = pd.to_datetime(daily_count['Date'])
        # Reset index to day of year.
        self.reset_daily_count_index(daily_count)
        rd['data'] = daily_count
        return rd

    @staticmethod
    def regularize_timeseries(rd):
        """Regularize count data.

        Copy's the object's count data and averages out observations with
        duplicate timestamps.

        Parameters
        ----------
        rd : RawCount
            Raw count data object.

        """

        crd = rd['data'].copy()

        def round_timestamp(timestamp):
            # timestamp must be pandas.Timestamp.
            seconds_after_hour = timestamp.minute * 60 + timestamp.second
            if seconds_after_hour % 900:
                ds = int(np.round(
                    seconds_after_hour / 900.)) * 900 - seconds_after_hour
                return timestamp + np.timedelta64(ds, 's')
            return timestamp

        crd['Timestamp'] = crd['Timestamp'].apply(round_timestamp)

        # If duplicate timestamps exist, use the arithmetic mean of the counts.
        # Regardless, convert counts to floating point.
        if crd['Timestamp'].duplicated().sum():
            # groupby sorts keys by default.
            crd = pd.DataFrame(
                {'Count': crd.groupby('Timestamp')['Count'].mean()})
            crd.reset_index(inplace=True)
        else:
            crd.sort_values('Timestamp', inplace=True)
            crd['Count'] = crd['Count'].astype(np.float64)

        # Required for calculating averaged data.
        crd['Date'] = crd['Timestamp'].dt.date

        return crd

    def get_zipreader(self, zipname):
        """Create an iterator over 15-minute count zip file.

        Assumes TEPs-I format: data is tab-delimited, and centreline
        ID is 2nd column and direction is 3rd column.
        """

        # Cycle through all files in the zip and append data to a list.
        with zipfile.ZipFile(zipname) as fhz:
            for fn in fhz.filelist:
                # Decode centreline ID and direction from first line of file.
                with fhz.open(fn) as fh:
                    first_line = fh.readline().decode('utf8').split('\t')
                    centreline_id = first_line[1]
                    direction = int(first_line[2])

                # Reopen file to read as a pandas DataFrame (can't seek because
                # zip decompression streams don't allow it).
                with fhz.open(fn) as fh:
                    data = pd.read_csv(fh, sep='\t', header=None,
                                       usecols=[3, 4], parse_dates=[0, ],
                                       infer_datetime_format=True)
                    data.columns = ['Timestamp', 'Count']

                # Check if file has enough data to return.
                # TO DO: log a warning if file is insufficient?
                if self.has_enough_data(data):
                    yield {'filename': fn.filename,
                           'centreline_id': int(centreline_id),
                           'direction': int(direction),
                           'data': data,
                           'year': data.at[0, 'Timestamp'].year}


class ReaderPostgres(ReaderBase):

    def read_source(self, counts):
        """Read PostgreSQL table contents into RawAnnualCount objects."""
        # Cycle through all relevant years.
        for year in range(cfg.cm['min_year'], cfg.cm['max_year'] + 1):
            current_counts = [
                RawAnnualCount.from_raw_data(
                    self.preprocess_count_data(c))
                for c in self.get_sqlreader(year)]

            # Append counts to processed count dicts.
            self.append_counts(current_counts, counts)

    def preprocess_count_data(self, rd):
        """Minor preprocessing of raw count data."""
        daily_count = rd['data'].copy()
        self.reset_daily_count_index(daily_count)
        rd['data'] = daily_count
        return rd

    def get_sqlreader(self, year):
        with self.source.connect() as db_con:
            sql_cmd = (
                ("SELECT centreline_id, direction, count_date, daily_count "
                 "FROM {dbt} WHERE count_year = {year} "
                 "ORDER BY centreline_id, direction, count_date")
                .format(dbt=self.source.tablename,  year=year))

            all_data = pd.read_sql(sql_cmd, db_con,
                                   parse_dates=['count_date', ])

            for key, df in all_data.groupby(['centreline_id', 'direction']):
                centreline_id = key[0]
                direction = key[1]

                data = df[['count_date', 'daily_count']].copy()
                data.columns = ['Date', 'Daily Count']

                # Filename is used to flag for HW401 data in Arman's zip files,
                # so just pass a dummy value here.  Note that we can't use
                # 'postgres' here since it contains 're'!
                yield {'filename': 'fromPG',
                       'centreline_id': int(centreline_id),
                       'direction': int(direction),
                       'data': data,
                       'year': year}


def read(source):

    rdr = (ReaderPostgres(source) if isinstance(source, conn.Connection)
           else ReaderZip(source))

    rdr.read()

    return rdr
