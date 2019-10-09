"""I/O classes for countmatch."""

import pandas as pd
import zipfile


class CountData:
    """Stores count data from a single traffic count file."""

    def __init__(self, centreline_id, direction, data):
        self.centreline_id = centreline_id
        self.direction = direction
        self.data = data


def read_zip(zipname, fast=True):
    """Read 15-minute count zip file into CountData objects.

    Assumes data is tab-delimited, and centreline ID is 2nd column and
    direction is 3rd column.
    """

    # Check if the file's actually a zip.
    if not zipfile.is_zipfile(zipname):
        raise IOError('{0} is not a zip file.'.format(zipname))

    # Cycle through all files in the zip.
    with zipfile.ZipFile(zipname) as fhz:
        count_data_list = []
        # For each file, retrieve data and append it to a list.
        for fn in fhz.filelist:
            with fhz.open(fn) as fh:
                # Decode centreline ID and direction from first line of file.
                first_line = fh.readline().decode('utf8').split('\t')
                centreline_id = first_line[1]
                direction = int(first_line[2])

            # Read in data as a pandas DataFrame (can't seek because zip
            # decompression streams don't allow it).
            with fhz.open(fn) as fh:
                data = pd.read_csv(fh, sep='\t', header=None,
                                   usecols=[3, 4], parse_dates=[0, ],
                                   infer_datetime_format=fast)
                data.columns = ['Timestamp', 'Count']

            count_data_list.append(
                CountData(centreline_id, direction, data))

    return count_data_list
