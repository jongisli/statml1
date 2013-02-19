#!/usr/bin/python
import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

img_format = 'png'

#precond: sigma is a symmetric, non-singular 3x3 matrix, mu is a 3-d vector
#returns: the pdf for the gaussian distribution N(mu,sigma)
def norm_pdf_multivariate(mu, sigma):
   sigma = np.matrix(sigma)
   
   inv = sigma.I
   det = np.linalg.det(sigma)
   
   norm_const = 1.0/ ( math.pow((2*np.pi),float(3)/2) * math.pow(det,1.0/2) )
   def pdf(x):
       x_mu=(np.matrix(x-mu))
       return norm_const * math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
   return pdf

#precond: image points to an image file supported by PIL of size 640x480
#returns: mean and covariance of the colors in region of the rectangle
#         ((149,263)(329,327))
def probability_model(image):
   orig = Image.open(image)
   crop = orig.crop((149,263,329,327)) #The training set
   data = np.transpose(list(crop.getdata()))
   mean = np.mean(data, axis=1)
   covariance = np.cov(data)
   return (mean, covariance)

#precond: image points to an image file supported by PIL of size 640x480
#         mean is a 3-d vector
#         covariance is a symmetric, non-singular 3x3 matrix
#postcond: files data/Z_[image].data and data/Z_trans_[image].data have
#          been created. The data in Z gives the probabilities associated
#          with the image, with respect to the mean and covariance arguments.
#          The data in Z_trans is the same as in Z, except the range of its
#          values has been changed to that of a gray-scale image (0-255)
def density_estimate(image,mean,covariance):
   im = Image.open(image)
   gauss = norm_pdf_multivariate(mean, covariance)

   #Mapping the image pixels to their gaussian probability density
   Z = [gauss(x) for x in im.getdata()] 

   #saving results to file
   file_Z = open('data/Z_'+image+'.data', 'w')
   for z in Z:
      file_Z.write("%s\n" % z)
   file_Z.close()
   
   #changing the density range to 0-255, with highest value at 255
   # and lowest at 0
   maxZ = max(Z)
   minZ = min(Z)
   OldRange = (maxZ - minZ)
   
   transformed_Z = [(((x - minZ) * 255) / OldRange) for x in Z]

   #Saving results to file
   file_Z = open('data/Z_trans_'+image+'.data', 'w')
   for z in transformed_Z:
      file_Z.write("%s\n" % z)
   file_Z.close()

#precond: image points to an image file supported by PIL of size 640x480
#         the file data/Z_trans_[image].data exists and contains floats
#postcond: a png file [image]_density.png has been created from the
#          grayscale data of Z_trans
def display_model(image):
   im = Image.new('L', Image.open(image).size)
   datafile = 'data/Z_trans_'+image+'.data'
   im.putdata(get_Z(datafile))

   #Here we visualize the image for the density, asked for in 1.9
   plt.imshow(im)
   plt.savefig(image[:-4]+'_density.%s' % img_format, format=img_format)
   plt.close()

#precond: datafile is a path to an existant file containing a float on
#         each line
#returns: the floats as a list
def get_Z(datafile):
   return [float(line.strip()) for line in open(datafile)]

#precond: image points to an image file supported by PIL of size 640x480
#         the file data/Z_[image].data exists and contains floats
#returns: the center of mass of the images probability density
def get_weighted_position(image):
   width, _ = Image.open(image).size
   datafile = 'data/Z_'+image+'.data'
   Z = get_Z(datafile)
   q_hat = sum([np.array([i % width, i / width]) * z \
                for i,z in enumerate(Z)]) / sum(Z)
   return q_hat

#precond: image points to an image file supported by PIL of size 640x480
#         the file data/Z_[image].data exists and contains floats
#postcond: the file [image]_weighted_pos.png now has the original image
#          together with its center of mass
def plot_weighted_position(image):
   mean_x,mean_y = get_weighted_position(image)
   _, axes = plt.subplots()
   im = plt.imread(image)

   axes.imshow(im)
   axes.autoscale(False)
   axes.scatter(mean_x,mean_y, s=50, c='green', marker='x', linewidth=2,
               label='$\hat{q}$')
   
   #Here we plot the average position on top of the image
   plt.legend(loc=0, scatterpoints = 1)
   plt.savefig(image[:-4]+'_weighted_pos.%s' % img_format, format=img_format)
   plt.close()
   
#precond: image points to an image file supported by PIL of size 640x480
#         the file data/Z_[image].data exists and contains floats
#returns: The spatial covariance of the image, with respect to
#         data/Z_[image].data
def spatial_covariance(image):
   width, _ = Image.open(image).size
   datafile = 'data/Z_'+image+'.data'
   Z = get_Z(datafile)
   x,y = get_weighted_position(image)

   #Compute the spatial covariance and return
   C = sum([(np.array([[i % width, i / width]]) - np.array([[x,y]])) * \
            (np.array([[i%width, i/width]]) - np.array([[x,y]])).T * z \
            for i,z in enumerate(Z)]) / float(sum(Z))
   return C

#precond: image points to an image file supported by PIL of size 640x480
#         the file data/Z_[image].data exists and contains floats
#postcond: The file [image]_and_contours.png now contains the
#          contour_plot of image
def contour_plot(image):
   width, height = Image.open(image).size
   _, axes = plt.subplots()
   im = plt.imread(image)
   axes.imshow(im)
   axes.autoscale(False)

   #Compute weighted average
   mean_x, mean_y = get_weighted_position(image)

   #compute covariance
   covariance = spatial_covariance(image)

   
   delta = 1
   x = np.arange(0, width, delta)
   y = np.arange(0, height, delta)
   X, Y = np.meshgrid(x, y)

   #Create a contour plot from the gaussian distribution defined by
   #the weighted position and spatial covariance previously computed
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

   #Visualize the result as a new image
   axes.contour(X, Y, Z)
   plt.legend(loc=0, scatterpoints = 1)
   plt.savefig(image[:-4]+'_and_contours.%s' % img_format, format=img_format)
   plt.close()

if  __name__ == "__main__":
   display_model('kande1.pnm')
   display_model('kande2.pnm')

   


