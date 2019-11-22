import numpy as np
import pandas as pd
import sklearn.neighbors as skln

from .. import cfg
from .. import conn


class NeighbourBase:

    def __init__(self, n_neighbours, ptc_ids):
        self._n_neighbours = n_neighbours
        self._ptc_ids = ptc_ids


class NeighbourLonLatBase(NeighbourBase):
    # TO DO: class methods need to be made robust against data with repeated
    # IDs or lat-lons, or NaNs.

    # TO DO: should we just calculate neighbours by default?

    _r_earth = 6.371e3
    _metric = None

    def __init__(self, source, n_neighbours, ptc_ids):
        super().__init__(n_neighbours, ptc_ids)

        if isinstance(source, conn.Connection):
            with source.connect() as db_con:
                self.data = pd.read_sql(
                    ("SELECT centreline_id, lon, lat FROM {dbt} "
                     "ORDER BY centreline_id".format(dbt=source.tablename)),
                    db_con)
                self.data['centreline_id'] = (self.data['centreline_id']
                                              .astype(int))
        else:
            self.data = pd.read_csv(source)
        self.data.columns = ['Centreline ID', 'Lon', 'Lat']

        # Handle index to centreline_id conversions.
        _idx_to_id = self.data['Centreline ID'].copy()
        self._id_to_idx = pd.Series(_idx_to_id.keys().values,
                                    index=_idx_to_id.values)

        # Create a permanent count-only frame
        self.df_ptc = self.data.loc[self.to_idxs(ptc_ids), :].copy()
        self.df_ptc.reset_index(drop=True, inplace=True)

    def lonlat_to_xy(self, lon, lat):
        """Converts long-lat coordinates to a physical coordinate system.

        The coordinate system is a linearization of a spherical shell, with
        an origin (0, 0) at the location of 'centre_of_toronto'.
        """
        lon = np.radians(lon)
        lat = np.radians(lat)
        lon0, lat0 = np.radians(cfg.distances['centre_of_toronto'])

        dlat = lat - lat0
        dlon = (lon - lon0) * np.cos(0.5 * (lat + lat0))

        # Convert to physical distances and rotate.
        return self._r_earth * np.c_[dlon, dlat]

    def get_xy(self, lon, lat):
        """Calculate Manhattan x/y."""
        # Convert latlon to physical distances.
        return self.lonlat_to_xy(lon, lat)

    def to_idxs(self, ids):
        return self._id_to_idx[ids].values

    def query_tree(self, X, y, n):
        """Build a tree to determine nearest points in y from points in X.

        Parameters
        ----------
        X : numpy.ndarray
            Array of points to query neighbours.  First column is longitude,
            second latitude.
        y : numpy.ndarray
            Array of neighbour points, with the same format as X.
        n : int
            Number of neighbours to retrieve.

        """
        btree = skln.BallTree(y, metric=self._metric)
        # This will retrieve itself, so ask for N_neighbours + 1.
        return btree.query(X, k=n)

    def get_neighbours(self):
        # Transform data to physical grid.
        X = self.get_xy(self.data['Lon'].values, self.data['Lat'].values)
        y = self.get_xy(self.df_ptc['Lon'].values, self.df_ptc['Lat'].values)
        # Query tree.
        n = min(self._n_neighbours + 1, y.shape[0])
        dists, idxs = self.query_tree(X, y, n)

        # Store neighbour data.
        neighbours = []
        distances = []
        for cidxs, cdists in zip(idxs, dists):
            # Return only non-zero neighbours.  If we need to return every
            # PTC in our data, allow PTCs to have one fewer neighbour than
            # STTCs.
            wanted = (slice(1, None, None) if cdists[0] == 0.
                      else slice(None, None if n == y.shape[0] else -1, None))
            neighbours.append(list(self.df_ptc['Centreline ID'][cidxs[wanted]]
                                   .values))
            distances.append(cdists[wanted])

        self.data['Neighbours'] = neighbours
        self.data['Distances'] = distances


class NeighbourLonLatEuclidean(NeighbourLonLatBase):

    _metric = skln.dist_metrics.EuclideanDistance()


class NeighbourLonLatManhattan(NeighbourLonLatBase):

    _metric = skln.dist_metrics.ManhattanDistance()

    def get_rotation_matrix(self):
        """2D rotation matrix.

        Uses **counterclockwise** rotation angle from configuration file.
        """
        gridangle = np.radians(cfg.distances['toronto_street_angle'])
        return np.array(
            [[np.cos(gridangle), -np.sin(gridangle)],
             [np.sin(gridangle), np.cos(gridangle)]])

    def get_xy(self, lon, lat):
        """Calculate Manhattan x/y."""
        # Convert latlon to physical distances.
        xy = self.lonlat_to_xy(lon, lat)
        # Rotate and sum to obtain Manhattan distances.  Transpose of the
        # usual rotation matrix, since we reverse the dot product order.
        return xy.dot(self.get_rotation_matrix())
