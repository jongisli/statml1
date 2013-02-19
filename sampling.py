#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from math import log
from scipy.stats import uniform

img_format = 'png'

# Sampling Methods

#precond: lambd is a positive real number
#         L is a natural number
#returns: average deviation after 1000
#         repetitions for simulating the
#         exponential distribution with
#         the formula:
#             y(x)=-(1/lambd)*log(1-x)
#         where x comes from the uniform
#         distribution. L controls the
#         amount of simulated values in
#         each repetition
def sampling_exponential_distribution(L, lambd):
    deviation = 0.0
    true_mean = 1 / lambd #mu_y
    repetitions = 1000
    for _ in range(repetitions):
        #invariant:
	#  deviation = sum of |mu_y-y_hat|
	#  for all y_hat values previously
	#  calculated
        
        #get L numbers from uniform dist
        z = uniform.rvs(size=L)

        #map uniform values to those given by expression y(x)=-(1/lambd)*log(1-x)
        y = map(lambda(x) : -(1/lambd)*log(1-x), z) 

	#calculate sample mean y_hat
        sample_mean = sum(y) / L

	#add new |mu_y-y_hat| to deviation
        deviation = deviation + abs(true_mean - sample_mean)
    
    #We return the average deviation of all
    #repetitions
    return deviation/repetitions

#precond: lambd is a positive real number
#postcond: montecarlo.png now contains a
#          picture of the convergence of
#          a simulated exponential
#          distribution
def convergence_plot(lambd):
    L = [10**i for i in range(5)]
    label = '$|\mu_y - \hat{y}|$'
    plt.plot(L, map(lambda(x) : sampling_exponential_distribution(x, lambd), L),
             label=label)
    plt.legend()
    axes = plt.axes()
    axes.set_xscale('log')
    plt.savefig("montecarlo.%s" % img_format, format=img_format)
    plt.close()

#precond: lambd is a positive real number
#postcond: montecarlo_logplot.png now contains a
#          picture of the logarithmic
#          convergence of a simulated
#          exponential distribution
def convergence_log_plot(lambd):
    L = [10**i for i in range(5)]
    label = '$\log(|\mu_y - \hat{y}|)$'
    plt.plot(L, map(lambda(x) : log(sampling_exponential_distribution(x, lambd)), L),
             label=label)
    plt.legend()
    axes = plt.axes()
    axes.set_xscale('log')
    plt.savefig("montecarlo_logplot.%s" % img_format, format=img_format)
    plt.close()

if __name__ == "__main__":
    convergence_log_plot(0.5)
