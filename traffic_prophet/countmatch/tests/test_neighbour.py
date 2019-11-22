import numpy as np
import sklearn.neighbors as skln

from .. import neighbour as nbr

from ...data import SAMPLE_LONLAT
from ... import cfg


class TestNeighbourLonLatEuclidean:
    """Tests Euclidean file-based neighbour finder.  Also tests base class."""

    def setup(self):
        # 890 and 104870 are permanent count locations.
        self.nle = nbr.NeighbourLonLatEuclidean(
            SAMPLE_LONLAT, 2, [890, 104870])

    def test_initialization(self):
        assert self.nle.data.shape == (11, 3)
        assert np.array_equal(self.nle.data.columns,
                              np.array(['Centreline ID', 'Lon', 'Lat']))

    def test_to_idxs(self):
        # Check that we can convert from IDs to indices.
        assert np.array_equal(
            self.nle.data.index.values,
            self.nle.to_idxs(self.nle.data['Centreline ID'].values))

        # Check that we can pass in single IDs.
        for cidx in np.random.choice(self.nle.data.index.values, size=10):
            assert cidx == self.nle.to_idxs(
                self.nle.data.at[cidx, 'Centreline ID'])

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

    def test_findneighbours(self):
        # Check that we're using the correct metric.
        assert isinstance(self.nle._metric,
                          skln.dist_metrics.EuclideanDistance)

        # Manually calculate pair-pair distances.
        xy = self.nle.get_xy(self.nle.data['Lon'].values,
                             self.nle.data['Lat'].values)
        distmtx = np.array(
            [((xy[:, 0] - xy[7, 0])**2 + (xy[:, 1] - xy[7, 1])**2)**0.5,
             ((xy[:, 0] - xy[9, 0])**2 + (xy[:, 1] - xy[9, 1])**2)**0.5]).T
        closest_arg = np.argsort(distmtx)
        distmtx_sorted = distmtx[
            np.arange(distmtx.shape[0], dtype=int)[:, np.newaxis],
            closest_arg]
        ids = np.array([890, 104870])

        # Run neighbour finder.
        self.nle.find_neighbours()

        # Check that the results are identical (take advantage of the fact that
        # index is just range(11)).
        for i in range(distmtx.shape[0]):
            # For PTCs don't compare the first value.
            s = 1 if i in (7, 9) else 0
            assert np.array_equal(self.nle.data.at[i, 'Neighbours'],
                                  ids[closest_arg[i, s:]])
            assert np.allclose(self.nle.data.at[i, 'Distances'],
                               distmtx_sorted[i, s:])

    def test_getneighbours(self):
        self.nle.find_neighbours()
        for i in range(self.nle.data.shape[0]):
            nbs, dists = self.nle.get_neighbours(
                self.nle.data.at[i, 'Centreline ID'])
            assert np.array_equal(nbs, self.nle.data.at[i, 'Neighbours'])
            assert np.array_equal(dists, self.nle.data.at[i, 'Distances'])


class TestNeighbourLonLatManhattan:

    def setup(self):
        # 890 and 104870 are permanent count locations.
        self.nlm = nbr.NeighbourLonLatManhattan(
            SAMPLE_LONLAT, 2, [890, 104870])

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
        distmtx = np.array(
            [np.abs(xy[:, 0] - xy[7, 0]) + np.abs(xy[:, 1] - xy[7, 1]),
             np.abs(xy[:, 0] - xy[9, 0]) + np.abs(xy[:, 1] - xy[9, 1])]).T
        closest_arg = np.argsort(distmtx)
        distmtx_sorted = distmtx[
            np.arange(distmtx.shape[0], dtype=int)[:, np.newaxis],
            closest_arg]
        ids = np.array([890, 104870])

        # Run neighbour finder.
        self.nlm.find_neighbours()

        # Check that the results are identical (take advantage of the fact that
        # index is just range(11)).
        for i in range(distmtx.shape[0]):
            # For PTCs don't compare the first value.
            s = 1 if i in (7, 9) else 0
            assert np.array_equal(self.nlm.data.at[i, 'Neighbours'],
                                  ids[closest_arg[i, s:]])
            assert np.allclose(self.nlm.data.at[i, 'Distances'],
                               distmtx_sorted[i, s:])
