import numpy as np
import pandas as pd
import sklearn.neighbors as skln
from .. import cfg


class NeighbourBase:

    def __init__(self, n_neighbours):
        self._n_neighbours = n_neighbours


class NeighbourLocalBase(NeighbourBase):

    _r_earth = 6.371e3
    _metric = None

    def __init__(self, sourcefile, n_neighbours):
        super().__init__(n_neighbours)

        self.data = pd.read_csv(sourcefile)
        self.data.columns = ['ID', 'Lon', 'Lat']
        self._id_to_idx = self.data['ID'].copy()
        # Technically pd.Series doesn't copy its initializing data, but
        # this isn't publicly accessible.
        self._idx_to_id = pd.Series(self._id_to_idx.keys().values,
                                    index=self._id_to_idx.values)

    def lonlat_to_xy(self, lon, lat):
        lat = np.radians(lat)
        lon = np.radians(lon)
        lon0, lat0 = np.radians(cfg.distances['centre_of_toronto'])

        dlat = lat - lat0
        dlon = (lon - lon0) * np.cos(0.5 * (lat + lat0))

        # Convert to physical distances and rotate.
        return self._r_earth * np.c_[dlon, dlat]

    def get_xy(self, lat, lon):
        """Calculate Manhattan x/y."""
        # Convert latlon to physical distances.
        return self.lonlat_to_xy(lat, lon)

    def to_ids(self, idxs):
        return self._id_to_idx[idxs].values

    def to_idxs(self, ids):
        return self._idx_to_id[ids].values

    def query_tree(self, xy):
        btree = skln.BallTree(xy, metric=self._metric)
        # This will retrieve itself, so ask for N_neighbours + 1.
        return btree.query(xy, k=(self._n_neighbours + 1))

    def get_neighbours(self):
        # Transform data to physical grid.
        xy = self.get_xy(self.data['Lat'].values,
                         self.data['Lon'].values)
        # Query tree.
        dists, idxs = self.query_tree(xy)

        # Convert indices to IDs and store.
        self.data['Neighbours'] = [self.to_ids(idx[1:]) for idx in idxs]

        # Store non-zero distances.
        self.data['Distances'] = [d[1:] for d in dists]


class NeighbourLocalEuclidean(NeighbourLocalBase):

    _metric = skln.dist_metrics.EuclideanDistance()


class NeighbourLocalManhattan(NeighbourLocalBase):

    _metric = skln.dist_metrics.ManhattanDistance()

    def get_rotation_matrix(self):
        gridangle = np.radians(cfg.distances['toronto_street_angle_degrees'])
        return np.array(
            [[np.cos(gridangle), np.sin(gridangle)],
             [-np.sin(gridangle), np.cos(gridangle)]])

    def get_xy(self, lat, lon):
        """Calculate Manhattan x/y."""
        # Convert latlon to physical distances.
        xy = self.lonlat_to_xy(lat, lon)
        # Rotate and sum to obtain Manhattan distances.  Transpose of the
        # usual rotation matrix, since we reverse the dot product order.
        return xy.dot(self.get_rotation_matrix())
