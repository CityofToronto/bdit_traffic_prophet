import numpy as np
import sklearn.metrics as sklm
import sklearn.neighbors as skln

from .. import neighbour as nbr

from ...data import SAMPLE_LONLAT
from ... import cfg


class TestNeighbourLonLatEuclidean:
    """Tests Euclidean file-based neighbour finder.  Also tests base class."""

    def setup(self):
        # There are only 10 items, so this orders all other points by distance.
        self.nle = nbr.NeighbourLonLatEuclidean(SAMPLE_LONLAT, 9)

    def test_initialization(self):
        assert self.nle.data.shape == (10, 3)
        assert np.array_equal(self.nle.data.columns,
                              np.array(['ID', 'Lon', 'Lat']))
        # Check that we can convert from indices to IDs and back again.
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

        assert np.allclose(outs, self.nle.get_xy(lons, lats))

    def test_getneighbours(self):
        # Check that we're using the correct metric.
        assert isinstance(self.nle._metric,
                          skln.dist_metrics.EuclideanDistance)

        # Manually calculate pair-pair distances.
        xy = self.nle.get_xy(self.nle.data['Lon'].values,
                             self.nle.data['Lat'].values)
        distmtx = sklm.pairwise_distances(xy, metric='euclidean')

        # Run neighbour finder.
        self.nle.get_neighbours()

        # Check that the results are identical (take advantage of the fact that
        # index is just range(10)).
        for i in range(distmtx.shape[0]):
            assert np.array_equal(
                self.nle.to_idxs(self.nle.data.at[i, 'Neighbours']),
                np.argsort(distmtx[i])[1:])
            assert np.sort(distmtx[i])[0] == 0.
            assert np.allclose(self.nle.data.at[i, 'Distances'],
                               np.sort(distmtx[i])[1:])

        # Just to show our estimate method is sensible, also check
        # haversine distances.
        hvrdistmtx = (
            sklm.pairwise.haversine_distances(
                np.radians(self.nle.data[['Lat', 'Lon']].values)) *
            self.nle._r_earth)
        for i in range(hvrdistmtx.shape[0]):
            assert np.array_equal(
                self.nle.to_idxs(self.nle.data.at[i, 'Neighbours']),
                np.argsort(hvrdistmtx[i])[1:])
            assert np.allclose(self.nle.data.at[i, 'Distances'],
                               np.sort(hvrdistmtx[i])[1:],
                               rtol=1e-3, atol=1e-5)


class TestNeighbourLonLatManhattan:

    def setup(self):
        # There are only 10 items, so this orders all other points by distance.
        self.nlm = nbr.NeighbourLonLatManhattan(SAMPLE_LONLAT, 9)

    def test_getxy(self):
        lons = self.nlm.data['Lon'].values
        lats = self.nlm.data['Lat'].values
        # Create a rotation matrix, and compare against the one from neighbour.
        theta = np.radians(cfg.distances['toronto_street_angle'])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        xy = self.nlm.lonlat_to_xy(lons, lats).dot(R)
        assert np.allclose(xy, self.nlm.get_xy(lons, lats))

    def test_getneighbours(self):
        # Check that we're using the correct metric.
        assert isinstance(self.nlm._metric,
                          skln.dist_metrics.ManhattanDistance)

        # Manually calculate pair-pair distances.
        xy = self.nlm.get_xy(self.nlm.data['Lon'].values,
                             self.nlm.data['Lat'].values)
        distmtx = sklm.pairwise_distances(xy, metric='manhattan')

        # Run neighbour finder.
        self.nlm.get_neighbours()

        # Check that the results are identical (take advantage of the fact that
        # index is just range(10)).
        for i in range(distmtx.shape[0]):
            assert np.array_equal(
                self.nlm.to_idxs(self.nlm.data.at[i, 'Neighbours']),
                np.argsort(distmtx[i])[1:])
            assert np.allclose(self.nlm.data.at[i, 'Distances'],
                               np.sort(distmtx[i])[1:])
