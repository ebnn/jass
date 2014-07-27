"""This module contains the main sampling classes."""

import numpy as np


class Sampler(object):
    """Abstract base class for all slice samplers."""

    def sample(self, logprob, init, rd):
        """Generate samples from a target distribution.

        Parameters
        ----------
        logprob : callable
            A function which takes in a parameter vector and returns the log of
            a value proportional to the probability density of the target
            distribution at that point.
        init : array-like
            The initial parameter vector of the chain.
        rd : `numpy.random.RandomState` instance
            An object representing the state of the pseudo-random number
            generator.
        """
        raise NotImplementedError()


class ComponentWiseSlice(Sampler):
    """Implementation of a component-wise slice sampler.

    This sampler considers each dimension in turn, and uses a univariate slice
    sampling procedure on each dimension (Neal 2003). This sampler uses the
    "step out" method for expanding the interval to cover the slice, with no
    limit on the size of the interval. It also includes an optional adaptive
    stage where the width in each dimension is calculated from the weighted
    average of the absolute difference in the samples. The samples generated
    from the adaptive stage are not used.

    Attributes
    ----------
    widths : array-like [ndims]
        The initial widths of the intervals in each dimension. If adaption is
        used, then after running the sampler, this becomes the adapted widths.
    ntune : int
        The number of adaptive steps to make per dimension.

    References
    ----------
    Neal R (2003). "Slice Sampling (with Discussion)." Annals of Statistics,
        31(3), 705-767.
    """

    def __init__(self, widths=None, ntune=0):
        """Create a new component-wise slice sampler.

        Parameters
        ----------
        widths : array-like [ndims], optional
            The initial widths of the intervals in each dimension. The width
            should be roughly the same as the expected size of a slice. It is
            generally better to overestimate this value than to underestimate.
            Either way, the sampler will only use these values as initial
            estimates and will expand and shrink intervals during runtime.
            By default, the widths are all set to 1.0.
        ntune : int, optional
            The number of adaptive steps to make in each dimension. If this
            is a positive integer, then the sampler will first adapt the widths
            in a pre-run, before collecting the actual samples. By default,
            this is set to 0 and no adaption is performed.
        """
        self.widths = widths
        self.ntune = ntune

    def sample(self, logprob, init, rd):
        # Set the default width to be 1 for all dimensions
        if self.widths is None:
            self.widths = np.ones((len(init),), dtype=float)

        # Generator of samples
        state = self._sample(logprob, init, rd)

        if self.ntune > 0:
            # Tune the width parameters for each dimension by finding the
            # weighted average of the distance moved from each state to the
            # next (this is usually the standard deviation of the distribution)
            prev = init
            for i in range(self.ntune):
                for d in range(len(init)):
                    # Use an exponential moving average
                    cur = next(state)
                    delta = np.abs(cur[d] - prev[d])
                    self.widths[d] = 0.1 * delta + 0.9 * self.widths[d]

                    prev = cur

        # Now start actually taking samples
        while True:
            yield next(state)

    def _sample(self, logprob, init, rd):
        ndims = len(init)

        # The current state of the chain
        x = np.array(init, dtype=float)
        x_prob = logprob(x)

        # Random number generators that we'll need
        rd_exp = rd.exponential
        rd_uniform = rd.uniform

        while True:
            for d in range(ndims):
                # Width of the interval
                width = self.widths[d]

                # 'Height' of the slice
                logy = x_prob - rd_exp() - 1e-9

                a = x[d] - rd_uniform() * width
                b = a + width

                # Save the value of the current state in this dimension.
                # This allows us to reuse 'x' to evaluate other states.
                x_p = x[d]

                # Increase the size of the interval until the interval contains
                # a full slice ("step out").
                x[d] = a
                while logprob(x) > logy:
                    x[d] -= width
                a = x[d]

                x[d] = b
                while logprob(x) > logy:
                    x[d] += width
                b = x[d]

                # Shrink the interval by rejection sampling
                while True:
                    x[d] = a + rd_uniform() * (b - a)
                    x_prob = logprob(x)

                    if x_prob > logy:
                        break
                    elif x[d] < x_p:
                        a = x[d]
                    else:
                        b = x[d]

                yield x.copy()