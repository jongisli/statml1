#!/usr/bin/python
import Image
import numpy as np
import math

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

def display_model(transformed_z, size):
   im = Image.new('L', size)
   im.putdata(transformed_z)
   im.show()

def get_Z(file):
   return [float(line.strip()) for line in open(file)]

def get_object_position(width):
   Z = get_Z('data/prob_Z')
   q_hat = sum([i*z for i,z in enumerate(Z)]) / sum(Z)
   x = q_hat % width
   y = q_hat / width
   return (x,y)


