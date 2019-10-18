import pytest
import numpy as np
import pandas as pd

from ...data import SAMPLE_ZIP
from .. import reader


class TestReader:

    def test_read_zip(self):
        # Also incidently tests CountData object, but that thing's too simple
        # for much to go wrong.

        counts = reader.read_zip(SAMPLE_ZIP)
        assert len(counts) == 8
        assert ([c.filename for c in counts] ==
                ['241_4372_2010.txt',
                 '252_4505_2010.txt',
                 '410_8108_2010.txt',
                 '427_8256_2010.txt',
                 '487_9229_2010.txt',
                 '890_17700_2010.txt',
                 '104870_11510_2010.txt',
                 '446378_2398_2010.txt'])
        assert ([c.centreline_id for c in counts] ==
                [241, 252, 410, 427, 487, 890, 104870, 446378])
        assert ([c.direction for c in counts] ==
                [-1 for i in range(len(counts))])
        assert ([c.data.shape for c in counts] ==
                [(288, 2), (96, 2), (96, 2), (96, 2), (288, 2),
                 (27072, 2), (30912, 2), (10752, 2)])
        assert np.array_equal(
            counts[2].data.dtypes.values,
            np.array([np.dtype('<M8[ns]'), np.dtype('int64')]))
        assert (counts[2].data.at[3, 'Timestamp'] ==
                pd.to_datetime('2010-06-09 00:45:00'))
        assert counts[2].data.at[10, 'Count'] == 0

        # Check that reading a non-zip raises an error.
        with pytest.raises(IOError):
            counts = reader.read_zip('./__init__.py')
