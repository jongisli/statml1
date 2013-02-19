#!/usr/bin/python
import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

img_format = 'png'

#precond: sigma is a non-singular 2x2 matrix
#returns: a function that takes a x-mu value and computes the probability density
def norm_pdf_multivariate(mu, sigma):
   sigma = np.matrix(sigma)
   
   inv = sigma.I
   det = np.linalg.det(sigma)
   if det == 0:
      raise NameError("The covariance matrix can't be singular")
   norm_const = 1.0/ ( math.pow((2*np.pi),float(3)/2) * math.pow(det,1.0/2) )
   return (lambda(x_mu): \
           norm_const * math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T)))

def multivariate_pdf(X,Y,mu, sigma):
   Z = mlab.bivariate_normal(
      X,
      Y,
      sigmax=sigma[0][0],
      sigmay=sigma[1][1],
      mux=mu[0],
      muy=mu[1],
      sigmaxy=sigma[0][1])
   return Z

def probability_model(image):
   orig = Image.open(image)
   crop = orig.crop((149,263,329,327))
   data = np.transpose(list(crop.getdata()))
   mean = np.mean(data, axis=1)
   covariance = np.cov(data)
   return (mean, covariance)

def density_estimate(image,mean,covariance):
   orig = Image.open(image)
   #gauss = norm_pdf_multivariate(mean, covariance)
   print (orig.getdata())
   return orig
   X,Y = np.meshgrid(x, y)
   Z = multivariate_pdf(X,Y,mean,covariance) 
   #Z = [gauss(np.matrix(x - mean)) for x in orig.getdata()]
   file_Z = open('data/Z_'+image+'.data', 'w')
   for z in Z:
      file_Z.write("%s\n" % z)
   file_Z.close()
   
   maxZ = max(Z)
   minZ = min(Z)
   OldRange = (maxZ - minZ)
   
   transformed_Z = [(((x - minZ) * 255) / OldRange) for x in Z]
   file_Z = open('data/Z_trans_'+image+'.data', 'w')
   for z in transformed_Z:
      file_Z.write("%s\n" % z)
   file_Z.close()
   return transformed_Z

def display_model(image):
   im = Image.new('L', Image.open(image).size)
   datafile = 'data/Z_trans_'+image+'.data'
   im.putdata(get_Z(datafile))
   im.save(image[:-4]+'_density.%s' % img_format, format=img_format)

def get_Z(datafile):
   return [float(line.strip()) for line in open(datafile)]

def get_weighted_position(image):
   width, _ = Image.open(image).size
   datafile = 'data/Z_'+image+'.data'
   Z = get_Z(datafile)
   q_hat = sum([np.array([i % width, i / width]) * z \
                for i,z in enumerate(Z)]) / sum(Z)
   return q_hat

def plot_weighted_position(image):
   mean_x,mean_y = get_weighted_position(image)
   _, axes = plt.subplots()
   im = plt.imread(image)

   axes.imshow(im)
   axes.autoscale(False)
   axes.scatter(mean_x,mean_y, s=50, c='green', marker='x', linewidth=2,
               label='$\hat{q}$')
   plt.legend(loc=0, scatterpoints = 1)
   plt.savefig(image[:-4]+'_weighted_pos.%s' % img_format, format=img_format)
   plt.close()
   

def spatial_covariance(image):
   width, _ = Image.open(image).size
   datafile = 'data/Z_'+image+'.data'
   Z = get_Z(datafile)
   x,y = get_weighted_position(image)
   C = sum([(np.array([[i % width, i / width]]) - np.array([[x,y]])) * \
            (np.array([[i%width, i/width]]) - np.array([[x,y]])).T * z \
            for i,z in enumerate(Z)]) / float(sum(Z))
   return C

def contour_plot(image):
   width, height = Image.open(image).size
   _, axes = plt.subplots()
   im = plt.imread(image)
   axes.imshow(im)
   axes.autoscale(False)
   mean_x, mean_y = get_weighted_position(image)
   covariance = spatial_covariance(image)
   delta = 1
   x = np.arange(0, width, delta)
   y = np.arange(0, height, delta)
   X, Y = np.meshgrid(x, y)
   Z = mlab.bivariate_normal(
      X,
      Y,
      sigmax=covariance[0][0],
      sigmay=covariance[1][1],
      mux=mean_x,
      muy=mean_y,
      sigmaxy=covariance[0][1])
   axes.scatter(mean_x,mean_y,s=50, c='green', marker='x',linewidth=2,
               label='$\hat{q}$')
   axes.contour(X, Y, Z)
   plt.legend(loc=0, scatterpoints = 1)
   plt.savefig(image[:-4]+'_and_contours.%s' % img_format, format=img_format)
   plt.close()

if  __name__ == "__main__":
  mean,covariance = probability_model('kande1.pnm')
  density_estimate('kande1.pnm',mean,covariance)
  #contour_plot('kande2.pnm')

   


