from .. import base


# from ...data import SAMPLE_ZIP
# @pytest.fixture(scope="module", autouse=True)
# def counts():
#     return reader.read(SAMPLE_ZIP)


class TestCount:

    def test_count(self):
        count = base.Count('test', 1, -1., None)
        assert count.count_id == 'test'
        assert count.centreline_id == 1
        assert count.direction == -1
        assert count.data is None
        assert not count.is_permanent
