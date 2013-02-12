#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

img_format = 'png'

# Question 1: Probability and Parameter Estimation

#question 1.1
def gaussian_plots():
    param = [(-1,1), (0,2), (2,3)]
    for i, params in enumerate(param):
        mean, std_dev = params
        range = np.arange(-4*std_dev + mean, 4*std_dev + mean, 0.001)
        plt_label = '$(\mu, \sigma) = (%d,%d)$' % (mean, std_dev)
        plt.plot(range, norm.pdf(range,mean,std_dev), label=plt_label)

    plt.legend()
    plt.savefig('gaussian.%s' % img_format, format=img_format)

def data(N):
    mean = [1,1]
    cov = [[0.3,0.2],[0.2,0.2]]
    y = np.random.multivariate_normal(mean,cov,N)
    return y

def mean(data):
    return np.mean(np.transpose(data), axis=1)

def covariance(data):
    return np.cov(np.transpose(data))

# question 1.3
def estimate_params(sample):
    x,y = zip(*sample)
    sample_mean = mean(sample)
    mean_x, mean_y = sample_mean
    sample_cov = covariance(sample)
    plt.scatter(x,y)
    plt.scatter(1,1,s=50, c='red',
                marker='x', label='$\mu$')
    plt.scatter(mean_x,mean_y,s=50, c='green',
                marker='x', label='$\mu_{ML}$')
    plt.legend()
    plt.savefig('scatter.%s' % img_format, format=img_format)
    return (sample_mean, sample_cov)

# Question 3: Non-parametric estimation - histograms

# question 1.5
def histogram_plot(data, bins):
    x1,x2 = zip(*data)
    
    plt.figure(1)
    plt.hist(x1, bins=bins, label='$p(x_1)$')
    plt.annotate('#bins = %d' % bins, xy=(0.01, 0.95), xycoords='axes fraction')
    plt.legend()
    plt.savefig('histogram1.%s' % img_format, format=img_format)

    plt.figure(2)
    plt.hist(x2, bins=bins,label='$p(x_2)$')
    plt.annotate('#bins = %d' % bins, xy=(0.01, 0.95), xycoords='axes fraction')
    plt.legend()
    plt.savefig('histogram2.%s' % img_format, format=img_format)


# question 1.6
def histogram_and_analytical_plot(data, bins):
    x1,_ = zip(*data)
    mean = 1
    std_dev = 0.3**(0.5)
    range = np.arange(-5*std_dev + mean, 5*std_dev + mean, 0.001)
    plt.hist(x1, bins=bins, normed=True)
    plt.plot(range, norm.pdf(range,mean,std_dev), linewidth=2)
    plt.savefig('hist_and_analytical.%s' % img_format, format=img_format)


# question 1.7
def histogram_plot_2d(data):
    x, y = zip(*data)
    
    plt.figure(1)
    plt.hist2d(x,y,bins=10)
    plt.annotate('#bins = 10', xy=(0.01, 0.95), xycoords='axes fraction',
                 color='white')
    plt.savefig('histogram3d_10bins.%s' % img_format, format=img_format)

    plt.figure(2)
    plt.hist2d(x,y,bins=15)
    plt.annotate('#bins = 15', xy=(0.01, 0.95), xycoords='axes fraction',
                 color='white')
    plt.savefig('histogram3d_15bins.%s' % img_format, format=img_format)

    plt.figure(3)
    plt.hist2d(x,y,bins=20)
    plt.annotate('#bins = 20', xy=(0.01, 0.95), xycoords='axes fraction',
                 color='white')
    plt.savefig('histogram3d_20bins.%s' % img_format, format=img_format)


histogram_plot_2d(data(100000))
