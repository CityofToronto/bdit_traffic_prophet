"""Base classes and functions for countmatch."""


class Count:
    """Base class for all count objects."""

    def __init__(self, count_id, centreline_id, direction,
                 data, is_permanent=False):
        self.count_id = count_id
        self.centreline_id = int(centreline_id)
        self.direction = int(direction)
        self.is_permanent = bool(is_permanent)
        self.data = data
