#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from math import log
from scipy.stats import uniform

img_format = 'png'

# Sampling Methods

def sampling_exponential_distribution(L, lambd):
    super_sam = 0.0
    for _ in range(1000):
        z = uniform.rvs(size=L)
        y = map(lambda(x) : -(1/lambd)*log(1-x), z)
        sample_mean = sum(y) / L
        true_mean = 1 / lambd
        super_sam = super_sam + sample_mean/1000.0
        
    return abs(true_mean - super_sam)

def convergence_plot(lambd):
    L = [10**i for i in range(5)]
    plt.plot(L, map(lambda(x) : sampling_exponential_distribution(x, lambd), L))
    axes = plt.axes()
    axes.set_xscale('log')
    plt.show()

convergence_plot(0.5)
