"""Sample files used by the test suite."""

# Use private names to avoid inclusion in the Sphinx documentation.
from os import path as _path


def _full_path(name, dirname=_path.dirname(_path.abspath(__file__))):
    return _path.join(dirname, name)


SAMPLE_ZIP = _full_path('15_min_counts_2010_neg_sample.zip')
"""Sample zip of 15-minute bin count files."""
