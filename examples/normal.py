"""This example samples from a simple bivariate normal distribution."""

import jass.mcmc as mcmc
import jass.samplers as samplers
import numpy as np
import scipy.stats as stats
import triangle
import matplotlib.pyplot as pl

# Define the log-likelihood function to be a bivariate normal
normal_rv = stats.multivariate_normal(cov=np.identity(2))

# Initialise the chain at the mean
initial = [0.0, 0.0]
sampler = samplers.ComponentWiseSlice()
samples = mcmc.run(sampler, normal_rv.logpdf, initial, 5000)

# Plot the the samples
triangle.corner(samples)
pl.show()
