import pytest
import numpy as np
import pandas as pd

from .. import neighbour as nbr

from ...data import SAMPLE_LONLAT
from ... import cfg


class TestNeighbourLocalEuclidean:
    """Tests Euclidean file-based neighbour finder.  Also tests base class."""

    def setup(self):
        self.nle = nbr.NeighbourLocalEuclidean(SAMPLE_LONLAT, 11)

    def test_initialization(self):
        assert self.nle.data.shape == (10, 3)
        assert np.array_equal(self.nle.data.columns,
                              np.array(['ID', 'Lon', 'Lat']))
        assert np.array_equal(
            self.nle.data.index.values,
            self.nle.to_idxs(self.nle.to_ids(self.nle.data.index.values)))

    def test_getxy(self):
        # Also implicitly tests lonlat_to_xy.
        lon0, lat0 = cfg.distances['centre_of_toronto']

        lons = np.array([lon0, lon0, lon0 - 1.])
        lats = np.array([lat0, lat0 + 1., lat0])

        dtr = np.radians(1.)
        outs = np.array(
            [[0., 0.],
             [0., self.nle._r_earth * dtr],
             [-self.nle._r_earth * dtr * np.cos(np.radians(lat0)), 0.]])
    
        assert np.allclose(outs, self.nle.lonlat_to_xy(lons, lats))

    def test_getneighbours(self):
        pass