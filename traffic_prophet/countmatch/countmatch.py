"""Base classes and functions for countmatch."""

import numpy as np
import pandas as pd

from . import reader
from .. import cfg


class Count:

    def __init__(self, centreline_id, direction, data):
        self.centreline_id = int(centreline_id)
        self.direction = int(direction)
        self.data = data


class DailyCount(Count):

    def __init__(self, centreline_id, direction, data,
                 is_permanent=False):
        super().__init__(centreline_id, direction, data)
        self.is_permanent = bool(is_permanent)

    @staticmethod
    def _round_timestamp(timestamp):
        # Rounds timestamp to nearest 15 minutes.
        seconds_after_hour = timestamp.minute * 60 + timestamp.second
        if seconds_after_hour % 900:
            ds = int(np.round(
                seconds_after_hour / 900.)) * 900 - seconds_after_hour
            return timestamp + np.timedelta64(ds)
        return timestamp

    @staticmethod
    def is_permanent_count(rc):
        """Checks if a count is permanent.

        Checks if count is a permanent traffic count.  Currently permanent
        count needs to have all 12 months and a sufficient number of total
        days represented, not be from HW401 and not be excluded by the user
        in the config file.

        Parameters
        ----------
        rc : RawCount
            Raw data read in by `countmatch.reader` functions.
        """
        n_available_months = rc.data.dt.month.unique().shape[0]
        if (n_available_months == 12 and
                (rc.data.shape[0] >= cfg.cm['min_permanent_stn_days'] * 96)):
            excluded_pos_files = (
                rc.direction == 1 and
                rc.centreline_id in cfg.cm['exclude_ptc_pos'])
            excluded_neg_files = (
                rc.direction == -1 and
                rc.centreline_id in cfg.cm['exclude_ptc_neg'])
            if (not excluded_pos_files and not excluded_neg_files and
                    're' not in rc.filename):
                return True
        return False

    @staticmethod
    def get_permanent_count_data(crd):
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

        crd['Month'] = crd['Timestamp'].dt.month
        crd['Day of Week'] = crd['Timestamp'].dt.dayofweek

        # Calculate MADT.
        crd_m = crd.groupby('Month')
        madt = pd.DataFrame({
            'MADT': 96. * crd_m['Count'].sum() / crd_m['Count'].count(),
            'Days in Month': crd_m['Timestamp'].min().dt.days_in_month}
        )

        # Calculate day-of-week of month ADT.
        crd_date = (crd.groupby(['Date', 'Month', 'Day of Week'])['Count']
                    .agg(['sum', 'count'])
                    .reset_index(level=(1,2)))
        # Drop any days with incomplete data.
        crd_date = crd_date.loc[crd_date['count'] == 96, :]
        crd_dom = crd_date.groupby(['Month', 'Day of Week'])
        domadt = (crd_dom['sum'].sum().unstack() /
                  crd_dom['sum'].count().unstack())
        # Determine DoM conversion factor.  (Uses a numpy broadcasting trick.)
        dom_factor = madt['MADT'].values / domadt

        # TO DO: A much simpler way to calculate domadt, consistent with MADT
        # above, would be: 
        #     crd_dom = crd.groupby(['Date', 'Month'])
        #     domadt = (96. * crd_dom['Count'].sum().unstack() /
        #               crd_dom['Count'].count().unstack())
        # We should consider using this instead.

        # AADT estimated directly from MADT -
        aadt = ((madt['MADT'] * madt['Days in Month']).sum() /
                madt['Days in Month'].sum())
        
        return {'MADT': madt, 'DoMADT': domadt,
                'DoM Factor': dom_factor, 'AADT': aadt}

    @classmethod
    def from_rawcount(cls, rc):
        # Copy file and round timestamps to the nearest 15 minutes.
        crd = rc.data.copy()
        crd['Timestamp'] = crd['Timestamp'].apply(cls._round_timestamp)

        # If duplicate timestamps exist, use the arithmetic mean of the counts.
        # Regardless, convert counts to floating point.
        if crd.data['Timestamp'].duplicated().sum():
            # groupby sorts keys by default.
            crd = (crd.groupby('Timestamp')['Count']
                   .mean())
            crd.reset_index(inplace=True)
        else:
            crd.sort_values('Timestamp', inplace=True)
            crd['Count'] = crd['Count'].astype(np.float64)

        # Group by date and obtain estimated daily traffic for each day.
        crd['Date'] = crd['Timestamp'].dt.date
        crdg = crd.groupby('Date')
        daily_count = pd.DataFrame({
            'Daily Count': 96. * crdg['Count'].sum() / crdg['Count'].count()}
        )

        # If count is permanent, do some further processing.
        if cls.is_permanent_count(rc):
            # Save to a new object.
            ptc_data = cls.get_permanent_count_data(crd)
            ptc_data['Daily Count'] = daily_count
            return cls(rc.centreline_id, rc.direction, ptc_data,
                       is_permanent=True)

        # Save to a new object.
        return cls(rc.centreline_id, rc.direction, daily_count)


def countmatch(source):
    """Calculates AADTs from counts.

    Parameters
    ----------
    source : str
        Path and filename of the zip file containing count data.
    """

    raw_counts = reader.read_zip(source)

    # Cycle through and process all counts.
    daily_counts = []
    for c in raw_counts:
        if c.data.shape[0] >= cfg.cm['min_stn_count']:
            daily_counts.append(DailyCount.from_rawcount(c))

    if not len(daily_counts):
        raise ValueError("no count file has more than the "
                         "minimum number of rows, {0}.  Check "
                         "input data.".format(cfg.cm['min_stn_count']))

    return daily_counts
