from typing import Protocol
from dataclasses import dataclass
from numpy.typing import ArrayLike
import scipy.stats as stats
import numpy as np



class Distribution(Protocol):

    def mean(self) -> float:
        """Mean of the given distribution."""

    def rvs(self, size: int) -> list[float]:
        """Sample variates from the distribution."""

    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
    
    def ppf(self, q: float) -> float:
        """Percent point function (inverse of cdf)."""


class PointwiseInterpolatedDist(stats.rv_continuous):
    """
    values: (pk, qk) where pk are percentiles between 0 and 1, and xk are
    the quantile values of each percentil. pk and qk must have the same shape.fcr
    """

    def __init__(self, values: tuple[ArrayLike, ArrayLike]):
        pk, qk = values
        assert len(pk) == len(qk)
        self.pk = pk
        self.qk = qk
 
        self.intervals = []
        self.density = []
        for i in range(1, len(qk)):
            interval = (qk[i-1], qk[i])
            self.intervals.append(interval)
            weight = pk[i] - pk[i-1]
            density =  weight / (qk[i] - qk[i - 1])
            self.density.append(density)  
        
        # prob mass of each interval
        self.width = 1 / len(self.intervals)  

    def mean(self):
        cum_ = 0.0
        for interval, prob in zip(self.intervals, self.density):
            cum_ = (interval[1] - interval[0]) / 2 * prob
        return cum_

    def rvs(self, size=1):
        def _sample_single_sample():
            # first sample an interval
            interval_idx = np.random.randint(0, len(self.intervals), 1)[0]
            interval = self.intervals[interval_idx]
            # inside this interval, sample uniformly
            return np.random.uniform(interval[0], interval[1], size=1)[0]

        samples = []
        for _ in range(size):
            samples.append(_sample_single_sample())

        assert len(samples) == size
        return np.array(samples)

    def cdf(self, x) -> float:
        if x < min(self.qk):
            return 0
        elif x > max(self.qk):
            return 1
        
        for i, interval in enumerate(self.intervals):
            left, right = interval                  
            if x < right and x >= left:
                # cumulated weights of previous intervals
                cum_weight = self.pk[i]
                # plus the cumulated weights at the current interval
                cum_weight += (x - left) / (right - left) * self.width
                return cum_weight

    def ppf(self, q) -> float:
        if q > 1 or q < 0:
            return np.nan
        
        for i, percentile in enumerate(self.pk):
            ...
        

