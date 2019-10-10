import pytest
import numpy as np

from ...data import SAMPLE_ZIP
from .. import io


class TestReadZip:

    def test_read_zip(self):
        # Also incidently tests CountData object, but that thing's too simple
        # for much to go wrong.

        counts = read_zip(SAMPLE_ZIP)
        assert len(data) == 8
        assert [count.centreline_id for count in counts]