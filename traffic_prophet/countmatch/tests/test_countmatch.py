import pytest
import hypothesis

import numpy as np
import pandas as pd
import datetime

from ...data import SAMPLE_ZIP
from .. import reader


@pytest.fixture(scope="module", autouse=True)
def counts():
    return reader.read_zip(SAMPLE_ZIP)
