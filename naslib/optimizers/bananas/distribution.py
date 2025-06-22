from typing import Protocol
from dataclasses import dataclass
import scipy.stats as stats
import random
import numpy as np

def get_quantile_levels(num_quantiles: int, log: bool =True):
    if num_quantiles > 100:
        raise NotImplementedError("num_quantiles should not be larger than 100.")
    
    levels = np.linspace(0, 1, num=num_quantiles + 1)[1:-1] # avoid get 0 and 1
    if log:
        print(f"Predicting {len(levels)} quantiles: {levels}.")
    return levels


class Distribution(Protocol):

    def mean(self) -> float:
        """Mean of the given distribution."""

    def rvs(self, size: int) -> list[float]:
        """Sample variates from the distribution."""

    def pdf(self, x: float) -> float:
        """Probability density function."""

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
    
    def pdf(self, x: float) -> float:
        return self.dist.pdf(x=x)

    def cdf(self, x: float) -> float:
        return self.dist.cdf(x=x)
       
    def ppf(self, q: float) -> float:
        return self.dist.ppf(q=q)

    def expected_gain(self, x, ei_factor: float = 5.0):
        mu = self.dist.mean()
        scaled_std = self.dist.std() / ei_factor
        gam = (mu - x) / scaled_std
        ei = (mu - x) * stats.norm.cdf(gam) + scaled_std * stats.norm.pdf(gam)
        return ei


@dataclass
class Interval:
    left: float
    right: float
    density: float
    prob: float
    cum_prob: float


class PointwiseInterpolatedDist:
    """
    values: (pk, qk) where pk are percentiles between 0 and 1, and xk are
    the quantile values of each percentil. pk and qk must have the same shape.fcr
    """

    def __init__(self, values: tuple[np.array, np.array]):
        pk, qk = values
        # Assume the population extremums are infinite values and extend quatiles
        avg_delta_q = max(np.diff(qk))
        pk = np.concatenate([np.array([0, 0.001]), pk, np.array([0.999, 1])])
        qk = np.concatenate([np.array([-np.finfo(np.float32).max, qk[0] - avg_delta_q]), qk, np.array([qk[-1] + avg_delta_q, np.finfo(np.float32).max])])

        assert len(pk) == len(qk)
        self.pk = pk
        self.qk = qk
 
        self.intervals = []
        for i in range(1, len(qk)):
            left = qk[i-1]
            right = qk[i]
            prob = pk[i] - pk[i-1]
            interval = Interval(left=left, right=right, prob=prob, density=prob / (right - left + np.finfo(float).eps), cum_prob=pk[i-1])
            self.intervals.append(interval)
    
    def mean(self):
        margin = 10
        x = np.linspace(self.intervals[0].right - margin, self.intervals[-1].left + margin, 1000)
        y = [self.pdf(x_i) * x_i for x_i in x]
        return np.trapz(y=y, x=x)
    
    def std(self):
        mu = self.mean()
        margin = 10
        x = np.linspace(self.intervals[0].right - margin, self.intervals[-1].left + margin, 1000)
        y = [self.pdf(x_i) * (x_i - mu)**2 for x_i in x]
        return np.sqrt(np.trapz(y=y, x=x))

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
            
    def pdf(self, x) -> float:
        for i, interval in enumerate(self.intervals):  
            left = interval.left
            right = interval.right             
            if x < right and x >= left:
                return interval.density

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
    
    def expected_gain(self, x):
        margin = 10
        x_array = np.linspace(x, self.intervals[-1].left + margin, 1000)
        y = [self.pdf(x_i) * (x_i - x) for x_i in x_array]
        return np.trapz(y=y, x=x_array)


