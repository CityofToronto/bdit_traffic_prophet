import operator

from .. import base


class TestCount:

    def test_count(self):
        count = base.Count('test', 1, -1., None)
        assert count.count_id == 'test'
        assert count.centreline_id == 1
        assert operator.index(count.direction) == -1
        assert count.data is None
        assert not count.is_permanent
