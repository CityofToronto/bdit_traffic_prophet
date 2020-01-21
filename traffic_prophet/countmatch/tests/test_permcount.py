import numpy as np
import pytest

from .. import reader
from .. import base
from .. import permcount as pc
from .. import derivedvals as dv
from .. import growthfactor as gf

from ...data import SAMPLE_ZIP


@pytest.fixture(scope='module')
def pcproc(cfgcm_test):
    dvc = dv.DerivedVals('Standard')
    gfc = gf.GrowthFactor('Composite')
    return pc.PermCountProcessor(dvc, gfc, cfg=cfgcm_test)


class TestPermCount:

    def test_permcount(self, sample_counts):
        ptc = pc.PermCount.from_count_object(sample_counts.counts[-104870],
                                             [2010, 2012])
        assert isinstance(ptc, pc.PermCount)
        assert ptc.is_permanent
        assert ptc.perm_years == [2010, 2012]

        with pytest.raises(AttributeError) as excinfo:
            ptc.growth_factor
        assert "PTC has not had its growth factor fit!" in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            ptc = pc.PermCount.from_count_object(
                sample_counts.counts[-104870], [])
        assert "cannot have empty perm_years" in str(excinfo.value)


class TestPermCountProcessor:

    def test_setup(self, pcproc):
        assert isinstance(pcproc.dvc, dv.DerivedValsStandard)
        assert isinstance(pcproc.gfc, gf.GrowthFactorComposite)
        # We passed a custom cfgcm with one excluded ID.
        assert pcproc.excluded_ids == [-446378, ]

    def test_partition(self, pcproc, sample_counts):
        ptc_oy_permyears = pcproc.partition_years(sample_counts.counts[-890])
        assert np.array_equal(ptc_oy_permyears, np.array([2010], dtype=int))
        ptc_my_permyears = pcproc.partition_years(
            sample_counts.counts[-104870])
        assert np.array_equal(ptc_my_permyears,
                              np.array([2010, 2012], dtype=int))

    def test_ptcs_sttcs(self, pcproc, cfgcm_test):
        # Can't use the fixture since this test will alter tcs.
        tcs = reader.read(SAMPLE_ZIP, cfg=cfgcm_test)
        pcproc.get_ptcs_sttcs(tcs)

        assert sorted(tcs.ptcs.keys()) == [-104870, -890]
        assert (sorted(tcs.sttcs.keys()) ==
                [-446378, -1978, -680, -487, -427, -410,
                 -252, -241, -170, 170, 104870])

        for ptc in tcs.ptcs.values():
            assert isinstance(ptc, pc.PermCount)
            assert (sorted(list(ptc.adts.keys())) ==
                    ['AADT', 'MADT'])
            assert (sorted(list(ptc.ratios.keys())) ==
                    ['D_ijd', 'DoM_ijd', 'N_avail_days'])
            assert abs(ptc.growth_factor) > 0.

        for sttc in tcs.sttcs.values():
            assert isinstance(sttc, base.Count)
