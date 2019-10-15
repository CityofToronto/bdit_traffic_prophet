"""Base classes and functions for countmatch."""

import numpy as np
import pandas as pd

from .. import cfg


class Count:

    def __init__(self, centreline_id, direction, data):
        self.centreline_id = int(centreline_id)
        self.direction = int(direction)
        self.data = data


class ADTCount(Count):

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
    def _is_permanent_count(rc, madt):
        # Checks if count is a permanent traffic count.  Currently permanent
        # count needs to have all 12 months and a sufficient number of total
        # days represented, not be from HW401 and not be excluded by the user
        # in the config file.
        if (rc.data.shape[0] >= cfg.cm['min_permanent_stn_days'] * 96):
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

        # Determine MADT.
        crd['Month'] = crd['Timestamp'].dt.month
        crd['Day of Year'] = crd['Timestamp'].dt.dayofyear
        crd['Day of Week'] = crd['Timestamp'].dt.dayofweek

        madt = pd.DataFrame({
            'counts': crd.groupby('Month')['Count'].sum(),
            'n_days': crd.groupby('Month')['Count'].count() / 96.}
        )
        madt['MADT'] = madt['counts'] / madt['n_days']

        # If count is permanent, do some further processing.
        if cls._is_permanent_count(rc, madt):
            # Save to a new object.
            return cls(rc.centreline_id, rc.direction, madt,
                       is_permanent=True)

        # Save to a new object.
        return cls(rc.centreline_id, rc.direction, madt)


# Cycle through and process all counts.
# for c in counts:
#     c.usable = (True if c.data.shape[0] >= cfg.cm['min_stn_count']
#                 else False)
# if sum([c.usable for c in counts]) == 0:
#     raise ValueError("no count file has more than the "
#                      "minimum number of rows, {0}.  Check "
#                      "input data.".format(cfg.cm['min_stn_count']))
