import pytest
import hypothesis

import numpy as np
import pandas as pd
import datetime

from ...data import SAMPLE_ZIP
from .. import reader
from .. import countmatch as cm


@pytest.fixture(scope="module", autouse=True)
def counts():
    return reader.read_zip(SAMPLE_ZIP)


class TestDailyCount:

    @pytest.mark.parametrize('filenumber', list(range())
    def test_round_timestamp(self, filenumber):
        dc = cm.DailyCount(0, 0, None)
        # Ensure randomly drawn data has same resolution as actual input data.
        t_pd = pd.to_datetime(t).round('1s')

        assert abs((t_rounded - t_pd) / np.timedelta64(1, 'm')) <= 7.5
