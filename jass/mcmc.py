"""This module contains useful functions for running and analysing MCMC."""

import numpy as np


def run(sampler, logprob, init, nsteps, rd=None):
    """Run a slice sampler for a specified number of steps.

    Parameters
    ----------
    sampler : instance of `Sampler` or one of its derived classes.
        The sampler to run.
    logprob : callable
        A function which takes in a parameter vector and returns the log of a
        value proportional to the probability density of the target distribution
        at that point.
    init : array-like
        The initial parameter vector of the chain.
    nsteps : int
        The number of samples to collect.
    rd : `numpy.random.RandomState` instance, optional
        An object representing the state of the pseudo-random number generator.
    """
    if rd is None:
        rd = np.random.RandomState()

    samples = np.empty((nsteps, len(init)), dtype=float)
    for i, sample in zip(range(nsteps), sampler.sample(logprob, init, rd)):
        samples[i] = sample
    return samples