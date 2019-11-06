import pytest
import hypothesis as hyp
import numpy as np
import pandas as pd

from ...data import SAMPLE_ZIP
from .. import reader
from .. import growthfactor as gf

class TestGrowthFactor:
    """Test growth factor calculation."""

    def setup(self):
        rdr = reader.Reader(SAMPLE_ZIP)
        rdr.read()

    @hyp.given(slp=hyp.strategies.floats(min_value=-2., max_value=2.))
    @hyp.settings(max_examples=30)
    def test_exponential_factor_fit(self, slp):
        # Create a generic exponential curve.
        x = np.linspace(1.5, 2.7, 100)
        y = np.exp(slp * x)
        result = gf.exponential_factor_fit(x, y,
                                           {"year": x[0], "aadt": y[0]})
        assert np.abs(result.params[0] - slp) < 0.01

    @hyp.given(slp=hyp.strategies.floats(min_value=-2., max_value=2.),
               y0=hyp.strategies.floats(min_value=-2., max_value=2.))
    @hyp.settings(max_examples=30)
    def test_linear_factor_fit(self, slp, y0):
        # Create a generic line.
        x = np.linspace(0.5, 3.7, 100)
        y = slp * x + y0
        result = gf.linear_factor_fit(x, y)
        assert np.abs(result.params[1] - slp) < 0.01
