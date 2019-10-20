"""Sample files used by the test suite."""

# Use private names to avoid inclusion in the Sphinx documentation.
from os import path as _path


def _full_path(name, dirname=_path.dirname(_path.abspath(__file__))):
    return _path.join(dirname, name)


SAMPLE_ZIP = {
    '2010': _full_path('15_min_counts_2010_neg_sample.zip'),
    '2011': _full_path('15_min_counts_2011_neg_sample.zip'),
    '2012': _full_path('15_min_counts_2011_neg_sample.zip')}
"""Sample zip of 15-minute bin count files from 2010, 2011 and 2012."""

SAMPLE_LONLAT = _full_path('count_lonlat.csv')
"""Sample zip of 15-minute bin count files."""
