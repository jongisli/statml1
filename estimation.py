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
    plt.close()

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
    plt.close()
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

    plt.close()


# question 1.6
def histogram_and_analytical_plot(data, bins):
    x1,_ = zip(*data)
    mean = 1
    std_dev = 0.3**(0.5)
    range = np.arange(-3.5*std_dev + mean, 3.5*std_dev + mean, 0.001)
    plt.hist(x1, bins=bins, normed=True, label='$p(x_1)$')
    plt.plot(range, norm.pdf(range,mean,std_dev), linewidth=2,
             label='$\mathcal{N}(x|1,0.3)$')
    plt.legend()
    plt.savefig('hist_and_analytical.%s' % img_format, format=img_format)
    plt.close()


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

    plt.close()


if __name__ == "__main__":
    data100 = data(100)
    data1000 = data(1000)
    data10000 = data(10000)
    
    histogram_plot_2d(data10000)
    
    """data = np.array([[ 0.72193061,  0.55650806],
       [ 0.71297482,  1.07638665],
       [ 0.0092223 ,  0.31207955],
       [ 1.26288325,  0.53376191],
       [ 0.37936011,  0.41056112],
       [ 0.45977093,  0.90105456],
       [ 1.21328513,  1.38968883],
       [ 1.79445033,  1.84837734],
       [ 1.92186304,  1.96301853],
       [ 0.15731425,  0.25997341],
       [ 0.62793672,  0.70533181],
       [ 0.99486393,  1.01006694],
       [ 1.8476812 ,  0.85853678],
       [ 0.14284534,  0.88913123],
       [ 0.97716776,  1.09018915],
       [ 0.97161201,  0.58900998],
       [ 1.73500472,  1.44673582],
       [ 0.9647784 ,  0.67339812],
       [-0.0737985 ,  0.29697366],
       [ 0.52818022,  0.17606585],
       [ 0.65179958,  0.52030596],
       [ 1.17995644,  0.97501462],
       [ 0.93388565,  1.11121914],
       [ 0.37516049,  0.47811406],
       [ 2.28856566,  1.92727012],
       [ 0.75213189,  0.60649678],
       [ 0.7318502 ,  1.03739217],
       [ 1.0689742 ,  1.78706917],
       [ 1.49248181,  1.68003937],
       [ 0.92929904,  0.76517611],
       [ 1.01305702,  0.87592965],
       [ 1.68066055,  1.74996003],
       [ 0.47905599,  0.8866    ],
       [ 1.34995982,  1.17785485],
       [ 0.14056147,  0.43742631],
       [ 1.14034598,  0.59048899],
       [ 0.81685558,  0.86283475],
       [ 0.77493163,  0.78814645],
       [ 0.78272775,  1.16437973],
       [-0.03891889,  0.39602963],
       [ 0.24695063,  0.68468749],
       [ 0.65705899,  1.19921722],
       [ 0.75625179,  0.51066502],
       [ 1.94282437,  2.00203365],
       [ 0.3120499 ,  0.73915184],
       [ 0.40481216,  0.47550674],
       [ 0.99378278,  1.40024804],
       [ 1.29908681,  1.40593826],
       [ 0.25305162,  0.00595536],
       [ 1.42712167,  1.18255341],
       [ 1.37167545,  1.50937496],
       [ 1.00078164,  1.08894058],
       [ 0.30535403,  0.49734938],
       [ 1.33500435,  0.96301662],
       [ 0.91019813,  0.81695105],
       [ 0.71188197,  1.17003723],
       [ 0.69000046,  1.04905076],
       [ 0.87673898,  0.42501247],
       [ 1.25174865,  0.93950734],
       [ 1.84500802,  1.21012506],
       [-0.11342139,  0.45389379],
       [ 0.50949098,  1.35315014],
       [ 1.66303637,  1.52663501],
       [ 1.25161124,  1.09516981],
       [ 1.03730311,  1.22388444],
       [ 1.3490821 ,  1.19041928],
       [-0.40000405,  0.21139045],
       [ 0.16126175,  0.24460852],
       [ 0.47175214,  0.4866838 ],
       [ 0.61041822,  0.39119264],
       [ 0.91784034,  1.15885657],
       [ 1.47795819,  1.29158918],
       [-0.14656642,  0.31257236],
       [ 1.3808889 ,  1.50453295],
       [ 1.37033896,  1.46412463],
       [-0.0744577 ,  0.63083995],
       [ 0.12192975,  0.47079   ],
       [ 1.74950922,  1.52973459],
       [ 1.41792453,  1.54522115],
       [ 0.07715062, -0.05034012],
       [ 1.83700728,  1.57702777],
       [ 0.94535332,  0.64067317],
       [ 1.58643266,  1.61009736],
       [ 0.8661403 ,  1.0710702 ],
       [ 1.20950009,  1.21510719],
       [-0.42484873,  0.12512812],
       [ 1.03568596,  0.68615155],
       [ 0.99644923,  1.05526292],
       [ 1.0486149 ,  0.90956878],
       [ 1.0250742 ,  0.81378442],
       [ 1.11217724,  1.29401667],
       [ 1.31009392,  1.38829251],
       [ 0.70994221,  0.98957382],
       [ 1.18102012,  1.26734074],
       [ 0.40896285,  0.27386058],
       [ 0.8875179 ,  0.71563476],
       [ 2.09703402,  2.06702828],
       [ 0.67191743,  0.80783912],
       [ 0.56457787,  0.44008001],
       [ 1.26197261,  0.61118427]])"""


