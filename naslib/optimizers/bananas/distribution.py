from typing import Protocol
from numpy.typing import ArrayLike
import scipy.stats as stats
import random
import numpy as np


class Distribution(Protocol):

    def mean(self) -> float:
        """Mean of the given distribution."""

    def rvs(self, size: int) -> list[float]:
        """Sample variates from the distribution."""

    def cdf(self, x: float) -> float:
        """Cumulative distribution function.
        
        x: value of a variate sampled from the distributrion.
        """
    
    def ppf(self, q: float) -> float:
        """Percent point function (inverse of cdf).
        
        q: a float between 0 and 1 representing quantile level 
        """

    def explected_gain(self, x: float) -> float:
        """Expected value of the distribution larger than a given value.
        
        x: value of a variate sampled from the distributrion.
        """

    
class GaussianDist:

    def __init__(self, loc, scale):
        self.dist = stats.norm(loc=loc, scale=scale)

    def mean(self) -> float:
        return self.dist.mean()

    def rvs(self, size: int) -> list[float]:
        return self.dist.rvs(size=size)

    def cdf(self, x: float) -> float:
        return self.dist.cdf(x=x)
       
    def ppf(self, q: float) -> float:
        return self.dist.ppf(q=q)

    def expected_gain(self, x, ei_factor: float = 5.0):
        scaled_std = self.dist.std() / ei_factor
        gam = (x - self.dist.mean()) / scaled_std
        ei = self.dist.mean() + scaled_std * (self.dist.pdf(gam) / (1 - self.dist.cdf(gam)))

        # gam = (self.dist.mean() - x) / scaled_std
        # ei = scaled_std * (gam * self.dist.cdf(gam) + self.dist.pdf(gam))
        return ei


        predictions = predictor.query([arch_encoding], info)
    #         mean = np.mean(predictions)
    #         std = np.std(predictions)
    #         factored_std = std / ei_calibration_factor
    #         max_y = ytrain.max()
    #         gam = (mean - max_y) / factored_std
    #         ei_value = factored_std * (gam * norm.cdf(gam) + norm.pdf(gam))
    #         return ei_value
        ...


class PointwiseInterpolatedDist(stats.rv_continuous):
    """
    values: (pk, qk) where pk are percentiles between 0 and 1, and xk are
    the quantile values of each percentil. pk and qk must have the same shape.fcr
    """

    def __init__(self, values: tuple[ArrayLike, ArrayLike], std: float):
        pk, qk = values
        assert len(pk) == len(qk)
        # For simplicity, for now we just assume the population extremums are half standard
        # deviation distance away from the sample extremums.
        qk[0] = qk[0] - std / 2 
        qk[-1] = qk[-1] + std / 2
        self.pk = pk
        self.qk = qk
 
        self.intervals = []
        self.densities = []
       
        for i in range(1, len(qk)):
            interval = (qk[i-1], qk[i])
            self.intervals.append(interval)
            weight = pk[i] - pk[i-1]
            left, right = interval
            density =  weight / (right - left + np.finfo(dtype=float).eps)
            self.densities.append(density)  
        
        # prob mass of each interval
        self.width = 1 / len(self.intervals)  

    def mean(self):
        cum_ = 0.0
        for interval in self.intervals:
            cum_ += (interval[1] + interval[0]) / 2 * self.width
        return cum_

    def rvs(self, size=1):
        def _sample_single_var():
            # first sample an interval
            interval_idx = np.random.randint(0, len(self.intervals), 1)[0]  # overweight extremes
            # interval_idx = random.choices(range(len(self.intervals)), weights=self.densities, k=1)[0]
            interval = self.intervals[interval_idx]
            # inside this interval, sample uniformly
            return np.random.uniform(interval[0], interval[1], size=1)[0]

        samples = []
        for _ in range(size):
            samples.append(_sample_single_var())

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
            if q <= percentile:
                interval = self.intervals[i-1]
                left, right = interval
                break
        
        ratio = (q - percentile) / self.width 
        return ratio * (right - left) + left
        

