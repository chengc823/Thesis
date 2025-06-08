from typing import Protocol
from dataclasses import dataclass
import scipy.stats as stats
import random
import numpy as np

def get_quantile_levels(num_quantiles: int):
    if num_quantiles > 100:
        raise NotImplementedError("num_quantiles should not be larger than 100.")
    
    levels = np.linspace(0, 1, num=num_quantiles + 1)
    # replace the first one and the last one 
    levels[0] = 0.001
    levels[-1] = 0.999
    return levels


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

@dataclass
class Interval:
    left: float
    right: float
    density: float
    prob: float
    cum_prob: float



class PointwiseInterpolatedDist(stats.rv_continuous):
    """
    values: (pk, qk) where pk are percentiles between 0 and 1, and xk are
    the quantile values of each percentil. pk and qk must have the same shape.fcr
    """

    def __init__(self, values: tuple[np.array, np.array]):
        pk, qk = values
        # Assume the population extremums are infinite values
        qk = np.insert(qk, 0, -np.finfo(np.float32).max)
        qk = np.append(qk, [np.finfo(np.float32).max])
        pk = np.insert(pk, 0, 0.0)
        pk = np.append(pk, [1.0])

        assert len(pk) == len(qk)
        self.pk = pk
        self.qk = qk
 
        self.intervals = []
        for i in range(1, len(qk)):
            left = qk[i-1]
            right = qk[i]
            prob = pk[i] - pk[i-1]
            interval = Interval(left=left, right=right, prob=prob, density=prob / (right - left), cum_prob=pk[i-1])
            self.intervals.append(interval)
     
    def mean(self):
        cum_ = 0.0
        for interval in self.intervals[1: -1]:     # drop intervals with infinite approximations for mean computation
            cum_ += (interval.right + interval.left) / 2 * interval.prob
        return cum_

    def rvs(self, size=1):
        def _sample_single_var():
            # first sample an interval
            weights = [interval.prob for interval in self.intervals]
            interval_idx = random.choices(range(len(self.intervals)), weights=weights, k=1)[0]
            interval = self.intervals[interval_idx]
            # inside this interval, sample uniformly
            return np.random.uniform(interval.left, interval.right, size=1)[0]

        samples = []
        for _ in range(size):
            samples.append(_sample_single_var())

        assert len(samples) == size
        return np.array(samples)

    def cdf(self, x) -> float:
        for i, interval in enumerate(self.intervals):  
            left = interval.left
            right = interval.right             
            if x < right and x >= left:
                # cumulated weights of previous intervals
                cum_weight = self.pk[i]
                # plus the cumulated weights at the current interval
                cum_weight += (x - left) / (right - left) * interval.prob
                return cum_weight

    def ppf(self, q) -> float:
        if q >= 1 or q <= 0:
            return np.nan
        
        for i in range(len(self.intervals) - 1):
            if q > self.intervals[i].cum_prob and q < self.intervals[i + 1].cum_prob:
                interval = self.intervals[i]
                break
            else:
                interval = self.intervals[-1]
        
        ratio = (q - interval.cum_prob) / interval.prob
        left = interval.left
        right = interval.right
        return left + ratio * (right - left) 
        

