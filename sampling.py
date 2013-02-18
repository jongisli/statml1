#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from math import log
from scipy.stats import uniform

img_format = 'png'

# Sampling Methods

def sampling_exponential_distribution(L, lambd):
    deviation = 0.0
    repetitions = 1000
    for _ in range(repetitions):
        z = uniform.rvs(size=L)
        y = map(lambda(x) : -(1/lambd)*log(1-x), z)
        sample_mean = sum(y) / L
        true_mean = 1 / lambd
        deviation = deviation + abs(true_mean - sample_mean)
        
    return deviation/repetitions

def convergence_plot(lambd):
    L = [10**i for i in range(5)]
    plt.plot(L, map(lambda(x) : sampling_exponential_distribution(x, lambd), L))
    axes = plt.axes()
    axes.set_xscale('log')
    plt.savefig("montecarlo.%s" % img_format, format=img_format)

if __name__ == "__main__":
    convergence_plot(0.5)
