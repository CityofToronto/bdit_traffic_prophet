"""Determine permanent count year-on-year growth factors."""

import numpy as np
import statsmodels.api as sm

from . import reader
from . import growthfactor as gf
from .. import cfg


class PermCount(reader.Count):
    """Class to hold permanent count data and calculate growth factors.

    Parameters
    ----------
    count_id : int
        ID of count.
    centreline_id : int
        Centreline ID associated with count.
    direction : +1 or -1
        Direction of traffic travel.
    data : pandas.DataFrame
        Raw daily traffic counts.
    perm_years : list
        List of years to use as permanent count.
    dv_processor : DerivedVal instance
        For imputation and derived properties.
    growth_factor : GrowthFactor instance
        For estimating growth factor.
    process: bool
        Process PTC for derived properties and growth rate.  Default: `True`.
    """

    def __init__(self, count_id, centreline_id, direction, data,
                 perm_years):
        data = {'Daily Count': data}
        super().__init__(count_id, centreline_id, direction, data,
                         is_permanent=True)
        self.perm_info = perm_years

    @classmethod
    def from_count_object(cls, tc, perm_years):
        return cls(tc.count_id, tc.centreline_id, tc.direction, tc.data,
                   perm_years)


class PermCountProcessor:
    """Class for processing a list of counts into PTCs and STTCs.

    Currently a permanent count needs to have at least one year with all 12
    months and a sufficient number of total days represented, not be from HW401
    and not be excluded by the user in the config file.
    """

    def __init__(self, dv_calc, gf_calc, cfg=cfg.cm):
        self.dvc = dv_calc
        self.gfc = gf_calc
        self.cfg = cfg
        # Obtain a list of count_ids that (according to TEPs-I) should not be
        # PTCs because they reduce the accuracy of CountMatch.
        self.excluded_ids = (self.cfg['exclude_ptc_pos'] +
                             [-id for id in self.cfg['exclude_ptc_neg']])

    def partition_years(self, tc):
        """Partition data by year, and determine when `tc` is a PTC.

        Determines which years of a count satisfy the TEPs permanent count
        requirements.

        Parameters
        ----------
        rd : reader.Count
            Candidate for permanent traffic count.

        Returns
        -------
        perm_years : list
            List of years that satisfy permanent count requirements.
        """
        # If count location on exclusion list, it's not a PTC.
        if tc.count_id in self.excluded_ids:
            return []

        # Get number of days and months in each year.
        mg = tc.data['Date'].dt.month.groupby(level=[0, ])
        counts_per_year = mg.count()
        n_unique_months = mg.apply(lambda x: x.unique().shape[0])

        perm_years = counts_per_year[
            (n_unique_months >= self.cfg['min_permanent_months']) &
            (counts_per_year >= self.cfg['min_permanent_days'])].index.values

        return perm_years

    def get_ptcs_sttcs(self, tcs):
        for tc in tcs.counts:
            perm_years = self.partition_years(tc)
            if len(perm_years):
                tcs.ptcs[tc.count_id] = (
                    PermCount.from_count_object(tc, perm_years))
                self.dvc.calculate(tc.ptcs[tc.count_id])
                self.gfc.fit_growth(tc.ptcs[tc.count_id])
            else:
                tcs.sttcs[tc.count_id] = tc


def get_ptcs_sttcs(tcs):
    dv_calc = SOME STUFF
    gf_calc = SOME STUFF
    ptcproc = PermCountProcessor(dv_calc, gf_calc)
    ptcproc.get_ptcs_sttcs(tcs)
