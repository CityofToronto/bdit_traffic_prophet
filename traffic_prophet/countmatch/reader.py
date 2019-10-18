"""I/O classes for countmatch."""

import pandas as pd
import zipfile


class Count:
    """Base class for all counts."""

    def __init__(self, centreline_id, direction, data):
        self.centreline_id = int(centreline_id)
        self.direction = int(direction)
        self.data = data


class RawCount(Count):
    """Stores raw data from a single traffic count file."""
    def __init__(self, centreline_id, direction, data, filename):
        super().__init__(centreline_id, direction, data)
        self.filename = filename


def read_zip(zipname):
    """Read 15-minute count zip file into Count objects.

    Assumes data is tab-delimited, and centreline ID is 2nd column and
    direction is 3rd column.
    """

    # Check if the file's actually a zip.
    if not zipfile.is_zipfile(zipname):
        raise IOError('{0} is not a zip file.'.format(zipname))

    # Cycle through all files in the zip and append data to a list.
    with zipfile.ZipFile(zipname) as fhz:
        counts = []
        for fn in fhz.filelist:
            filename = fn.filename
            # Decode centreline ID and direction from first line of file.
            with fhz.open(fn) as fh:
                first_line = fh.readline().decode('utf8').split('\t')
                centreline_id = first_line[1]
                direction = int(first_line[2])

            # Reopen file to read as a pandas DataFrame (can't seek because zip
            # decompression streams don't allow it).
            with fhz.open(fn) as fh:
                data = pd.read_csv(fh, sep='\t', header=None,
                                   usecols=[3, 4], parse_dates=[0, ],
                                   infer_datetime_format=True)
                data.columns = ['Timestamp', 'Count']

            counts.append(
                RawCount(centreline_id, direction, data, filename))

    # Sort list by centreline ID.
    counts.sort(key=lambda cd: cd.centreline_id)
    return counts
