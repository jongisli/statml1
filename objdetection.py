#!/usr/bin/python
import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

img_format = 'png'

def norm_pdf_multivariate(mu, sigma):
   sigma = np.matrix(sigma)
   
   inv = sigma.I
   det = np.linalg.det(sigma)
   if det == 0:
      raise NameError("The covariance matrix can't be singular")
   norm_const = 1.0/ ( math.pow((2*np.pi),float(3)/2) * math.pow(det,1.0/2) )
   return (lambda(x_mu): \
           norm_const * math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T)))

def probability_model(image):
   orig = Image.open(image)
   crop = orig.crop((149,263,329,327))
   data = np.transpose(list(crop.getdata()))
   mean = np.mean(data, axis=1)
   covariance = np.cov(data)
   gauss = norm_pdf_multivariate(mean, covariance)
   Z = [gauss(np.matrix(x - mean)) for x in orig.getdata()]
   file_Z = open('data/prob_Z', 'w')
   for z in Z:
      file_Z.write("%s\n" % z)
   file_Z.close()
   
   maxZ = max(Z)
   minZ = min(Z)
   OldRange = (maxZ - minZ)
   
   transformed_Z = [(((x - minZ) * 255) / OldRange) for x in Z]
   file_Z = open('data/prob_Z_trans', 'w')
   for z in transformed_Z:
      file_Z.write("%s\n" % z)
   file_Z.close()
   return (mean, covariance)

def display_model(transformed_z, size):
   im = Image.new('L', size)
   im.putdata(transformed_z)
   im.show()

def get_Z(file):
   return [float(line.strip()) for line in open(file)]

def get_object_position(width):
   Z = get_Z('data/prob_Z')
   q_hat = sum([np.array([i % width, i / width]) * z \
                for i,z in enumerate(Z)]) / sum(Z)
   return q_hat

def plot_weighted_position(width):
   mean_x,mean_y = get_object_position(width)
   _, axes = plt.subplots()
   im = plt.imread("kande1.JPG")

   axes.imshow(im)
   axes.autoscale(False)
   axes.scatter(mean_x,mean_y, s=50, c='green', marker='x', linewidth=2,
               label='$\hat{q}$')
   plt.legend(loc=0, scatterpoints = 1)
   plt.savefig('kande1_weighted_pos.%s' % img_format, format=img_format)
   plt.close()
   

def spatial_covariance(width):
   Z = get_Z('data/prob_Z')
   x,y = get_object_position(width)
   C = sum([(np.array([[i % width, i / width]]) - np.array([[x,y]])) * \
            (np.array([[i%width, i/width]]) - np.array([[x,y]])).T * z \
            for i,z in enumerate(Z)]) / float(sum(Z))
   return C

def contour_plot(height,width):
   _, axes = plt.subplots()
   im = plt.imread("kande1.JPG")
   axes.imshow(im)
   axes.autoscale(False)
   mean_x, mean_y = get_object_position(width)
   covariance = spatial_covariance(width)
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
   C = axes.contour(X, Y, Z, label='probability contours')
   plt.legend(loc=0, scatterpoints = 1)
   plt.savefig('kande1_and_contours.%s' % img_format, format=img_format)
   plt.close()

if  __name__ == "__main__":
   contour_plot(640,480)


   


