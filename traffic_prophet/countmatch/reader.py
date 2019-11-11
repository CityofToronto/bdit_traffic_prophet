"""I/O and pre-processing classes for countmatch."""

import numpy as np
import pandas as pd
import zipfile
import glob
import warnings

from .. import cfg
from .. import conn


class Count:

    def __init__(self, centreline_id, direction, data, is_permanent=False):
        self.centreline_id = int(centreline_id)
        self.direction = int(direction)
        self.is_permanent = bool(is_permanent)
        self.data = data


class AnnualCount(Count):
    """Storage for total and average daily traffic."""

    def __init__(self, centreline_id, direction, year,
                 data, is_permanent=False):
        self.year = int(year)
        super().__init__(centreline_id, direction, data,
                         is_permanent=is_permanent)

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
        # These are only used if the count is from a permanent station, but
        # leaving here so that `crd` is idempotent within those functions.
        crd['Month'] = crd['Timestamp'].dt.month
        crd['Day of Week'] = crd['Timestamp'].dt.dayofweek

        return crd

    @staticmethod
    def process_15min_count_data(crd):
        """Calculates total daily traffic from raw count data."""
        crdg = crd.groupby('Date')
        daily_count = pd.DataFrame({
            'Daily Count': 96. * crdg['Count'].sum() / crdg['Count'].count()})
        daily_count.reset_index(inplace=True)
        daily_count['Date'] = pd.to_datetime(daily_count['Date'])
        return daily_count

    @staticmethod
    def reset_daily_count_index(daily_count):
        daily_count.set_index(
            pd.Index(daily_count['Date'].dt.dayofyear, name='Day of Year'),
            inplace=True)

    @staticmethod
    def is_permanent_count(rd):
        """Check if a count is permanent.

        Checks if count is a permanent traffic count.  Currently permanent
        count needs to have all 12 months and a sufficient number of total
        days represented, not be from HW401 and not be excluded by the user
        in the config file.

        Parameters
        ----------
        rd : dict
            Raw data, with entries for 'centreline_id', 'direction',
            'year' and 'data'.  The first three are integers, while 'data'
            is a `pandas.DataFrame` with 'Timestamp' and 'Count' columns.

        """
        n_available_months = rd['data']['Timestamp'].dt.month.unique().shape[0]
        if (n_available_months == 12 and
                (rd['data'].shape[0] >=
                 cfg.cm['min_permanent_stn_days'] * 96)):
            excluded_pos_files = (
                rd['direction'] == 1 and
                rd['centreline_id'] in cfg.cm['exclude_ptc_pos'])
            excluded_neg_files = (
                rd['direction'] == -1 and
                rd['centreline_id'] in cfg.cm['exclude_ptc_neg'])
            if (not excluded_pos_files and not excluded_neg_files and
                    're' not in rd['filename']):
                return True
        return False

    @staticmethod
    def process_permanent_count_data(crd):
        """Derives AADT, DoMADT and DoM factor values for permanent counts.

        Notes
        -----
        Averaging methods are identical to `STTC_estimate3.m` in the original
        TEPs-I.  To be entirely self-consistent, time bins with incomplete
        data (eg. months where only certain days are covered, days where
        only certain 15-minute segments) should either be partly retained (eg.
        by simply averaging all time bins of a certain month, or day of week
        within the month, together), completely dropped or inflated.  These all
        introduce their own biases, though those may be minor.

        Parameters
        ----------
        crd : pandas.DataFrame
            Data from a RawCount object processed within `from_rawcount`.

        """

        # Calculate MADT.
        crd_m = crd.groupby('Month')
        madt = pd.DataFrame({
            'MADT': 96. * crd_m['Count'].sum() / crd_m['Count'].count(),
            'Days in Month': crd_m['Timestamp'].min().dt.days_in_month}
        )

        # Calculate day-of-week of month ADT.
        crd_date = (crd.groupby(['Date', 'Month', 'Day of Week'])['Count']
                    .agg(['sum', 'count'])
                    .reset_index(level=(1, 2)))
        # Drop any days with incomplete data.
        crd_date = crd_date.loc[crd_date['count'] == 96, :]
        crd_dom = crd_date.groupby(['Month', 'Day of Week'])
        domadt = (crd_dom['sum'].sum().unstack() /
                  crd_dom['sum'].count().unstack())
        # Determine DoM conversion factor.  (Uses a numpy broadcasting trick.)
        dom_factor = madt['MADT'].values[:, np.newaxis] / domadt

        # TO DO: A much simpler way to calculate domadt, consistent with MADT
        # above, would be:
        #     crd_dom = crd.groupby(['Date', 'Month'])
        #     domadt = (96. * crd_dom['Count'].sum().unstack() /
        #               crd_dom['Count'].count().unstack())
        # We should consider using this instead.

        # Weighted average for AADT.
        aadt = np.average(madt['MADT'], weights=madt['Days in Month'])

        return {'MADT': madt, 'DoMADT': domadt,
                'DoM Factor': dom_factor, 'AADT': aadt}

    @classmethod
    def from_raw_data(cls, rd):
        """Processes data from 15-minute bin zip files.

        Parameters
        ----------
        rd : dict
            Raw data, with entries for 'centreline_id', 'direction',
            'year' and 'data'.  The first three are integers, while 'data'
            is a `pandas.DataFrame` with 'Count' column and either 'Timestamp'
            or 'Date' column.
        """
        # If we're reading from raw zip files.
        if 'Timestamp' in rd['data'].keys():
            # Copy file and round timestamps to the nearest 15 minutes.
            crd = cls.regularize_timeseries(rd)
            # Get daily total count values.
            daily_count = cls.process_15min_count_data(crd)
        # If we're instead reading from Postgres.
        elif 'Date' in rd.data.keys():
            daily_count = rd.data.copy()
        else:
            raise ValueError("raw input data missing "
                             "'Timestamp' or 'Date' column.")

        # Reset daily_count index, common to both zip files and Postgres.
        cls.reset_daily_count_index(daily_count)

        # If count is permanent, also get MADT, AADT, DoMADT and DoM factors.
        if cls.is_permanent_count(rd):
            # Save to a new object.
            ptc_data = cls.process_permanent_count_data(crd)
            ptc_data['Daily Count'] = daily_count
            return cls(rd['centreline_id'], rd['direction'], rd['year'],
                       ptc_data, is_permanent=True)

        # Save to a new object.
        return cls(rd['centreline_id'], rd['direction'], rd['year'],
                   daily_count)


class Reader:

    _sql_cmd = ("SELECT centreline_id, direction, count_date, daily_count "
                "FROM {dbt} WHERE count_year = {year} "
                "ORDER BY centreline_id, direction, count_date")

    def __init__(self, source):
        # Store permanent and temporary stations.
        self.ptcs = None
        self.sttcs = None

        # Handle read source.
        if isinstance(source, conn.Connection):
            # TO DO: figure out psycopg2 connection.
            self.sourcetype = 'SQL'
            self._reader = self.read_sql
        else:
            self.sourcetype = 'Zip'
            self._reader = self.read_zip
            if type(source) == str:
                source = glob.glob(source)
            elif type(source) == dict:
                source = [source[k] for k in sorted(source.keys())]
            source = sorted(source)

        self.source = source

    def read(self):
        """Read source data into dictionaries of counts."""
        self._reader()

    def read_zip(self):
        """Read zip file contents into AnnualCount objects."""
        # ptcs and sttcs hold arrays of processed counts.
        ptcs = {}
        sttcs = {}

        # Cycle through all zip files.
        for zf in self.source:
            # Check if the file's actually a zip.
            if not zipfile.is_zipfile(zf):
                raise IOError('{0} is not a zip file.'.format(zf))

            current_counts = [AnnualCount.from_raw_data(c)
                              for c in self.get_zipreader(zf)]

            # Append counts to processed count dicts.
            self.append_counts(current_counts, ptcs, sttcs)

        self.check_processed_count_integrity(ptcs, sttcs)
        self.unify_counts(ptcs, sttcs)
        self.ptcs = ptcs
        self.sttcs = sttcs

    def get_zipreader(self, zipname):
        """Create an iterator over 15-minute count zip file.

        Assumes Arman's TEPs-I format: data is tab-delimited, and centreline
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

    def read_sql(self):
        """Read PostgreSQL table contents into AnnualCount objects."""
        # ptcs and sttcs hold arrays of processed counts.
        ptcs = {}
        sttcs = {}

        # Cycle through all zip files.
        for year in range(cfg.cm['min_year'], cfg.cm['max_year'] + 1):
            current_counts = [AnnualCount.from_raw_data(c)
                              for c in self.get_sqlreader(year)]

            # Append counts to processed count dicts.
            self.append_counts(current_counts, ptcs, sttcs)

        # self.check_processed_count_integrity(ptcs, sttcs)
        # self.unify_counts(ptcs, sttcs)
        self.ptcs = ptcs
        self.sttcs = sttcs

    def get_sqlreader(self, year):
        with self.source.connect() as db_con:
            all_data = pd.read_sql(
                self._sql_cmd.format(dbt=self.source.tablename,  year=year),
                db_con, parse_dates=['count_date', ])

            for key, df in all_data.groupby(['centreline_id', 'direction']):
                centreline_id = key[0]
                direction = key[1]

                data = df[['count_date', 'daily_count']].copy()
                data.columns = ['Date', 'Daily Count']

                # Filename is used to flag for HW401 data in Arman's zip files,
                # so just pass a dummy value here.
                yield {'filename': 'frompostgres',
                       'centreline_id': int(centreline_id),
                       'direction': int(direction),
                       'data': data,
                       'year': year}

    @staticmethod
    def has_enough_data(data):
        """Checks if there is enough data to be a usable short count count."""
        return data.shape[0] >= cfg.cm['min_stn_count']

    @staticmethod
    def append_counts(current_counts, ptcs, sttcs):
        for c in current_counts:
            _appendto = ptcs if c.is_permanent else sttcs
            if c.centreline_id in _appendto.keys():
                _appendto[c.centreline_id].append(c)
            else:
                _appendto[c.centreline_id] = [c, ]

    @staticmethod
    def check_processed_count_integrity(ptcs, sttcs):
        if not len(sttcs) + len(ptcs):
            raise ValueError(
                "no count file has more than cfg.cm['min_stn_count'] == {0}."
                "  Check input data.".format(cfg.cm['min_stn_count']))
        elif not len(ptcs):
            warnings.warn(
                "no permanent counts read!  Check 'min_permanent_stn_days', "
                "'exclude_ptc_pos' and 'exclude_ptc_neg' settings, and "
                "confirm that counts covering all 12 months of a year exist.")
        elif not len(sttcs):
            warnings.warn("file only contains permanent counts!")

    @staticmethod
    def unify_counts(ptcs, sttcs):
        """Unify count objects across years.

        For each centreline ID in `ptcs`, create data tables for each
        value and all years.  Works **in place**, and replaces the index of
        each table with a MultiIndex.
        """
        # For permanent counts, perform in-place reindexing, then unify each
        # type of data into a dict.
        for cid in ptcs.keys():
            for item in ptcs[cid]:
                # Create multi-indexes for data in each count object.  This
                # action is NOT idempotent, but I prefer this to making copies
                # of tables.
                for subkey in ['MADT', 'DoMADT', 'DoM Factor', 'Daily Count']:
                    _ctable = item.data[subkey]
                    _ctable.index = pd.MultiIndex.from_product(
                        [[item.year, ], _ctable.index],
                        names=['Year', _ctable.index.name])
            unified_data = {
                'MADT': pd.concat([c.data['MADT'] for c in ptcs[cid]]),
                'DoMADT': pd.concat([c.data['DoMADT'] for c in ptcs[cid]]),
                'DoM Factor': pd.concat(
                    [c.data['DoM Factor'] for c in ptcs[cid]]),
                'Daily Count': pd.concat(
                    [c.data['Daily Count'] for c in ptcs[cid]]),
                'AADT': pd.DataFrame(
                    {'AADT': [c.data['AADT'] for c in ptcs[cid]]},
                    index=pd.Index([c.year for c in ptcs[cid]], name='Year'))}
            # Replace list of multipe counts with a single Count object.
            ptcs[cid] = Count(ptcs[cid][0].centreline_id,
                              ptcs[cid][0].direction,
                              unified_data, is_permanent=True)
        for cid in sttcs.keys():
            for item in sttcs[cid]:
                _ctable = item.data
                _ctable.index = pd.MultiIndex.from_product(
                    [[item.year, ], _ctable.index],
                    names=['Year', _ctable.index.name])
            unified_data = pd.concat([c.data for c in sttcs[cid]])
            sttcs[cid] = Count(sttcs[cid][0].centreline_id,
                               sttcs[cid][0].direction,
                               unified_data, is_permanent=False)
