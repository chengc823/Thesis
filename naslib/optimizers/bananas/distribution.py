from abc import abstractmethod, ABC
import scipy.stats as stats
import numpy as np


class Distribution(ABC):

    @abstractmethod
    def mean(self) -> float:
        """Mean of the given distribution."""

    @abstractmethod
    def rvs(self, size: int) -> list[float]:
        """Sample variates from the distribution."""

    @abstractmethod
    def cdf(self, percentile: float) -> float:
        """Cumulative distribution function."""
    
    @abstractmethod
    def ppf(self, quantile: float) -> float:
        """Percent point function (inverse of cdf)."""
