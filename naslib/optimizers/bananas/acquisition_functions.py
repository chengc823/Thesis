from naslib.optimizers.bananas.distribution import Distribution


def idependent_thompson_sampling(distribution: Distribution, **kwargs) -> float:
    return distribution.rvs(size=1)[0]


def upper_confidence_bound(distribution: Distribution, explore_factor: float = 0.5, **kwargs) -> float:
    """
    explore_factor: The percentile at which the quantile value is estimated.
    """
    return distribution.ppf(q=explore_factor)


def probability_of_improvement(distribution: Distribution, threshold: float, **kwags) -> float:
    return 1 - distribution.cdf(x=threshold)


def exploit_only(distribution: Distribution, **kwargs) -> float:
    return distribution.mean()


def expected_improvement(distribution: Distribution, threshold: float, **kwargs) -> float:
    return distribution.expected_gain(x=threshold, **kwargs)