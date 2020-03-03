# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:24:25 2020

@author: cyber-wiz
"""
# Edge Detection
#from skimage import data,img_as_float

#from skimage.morphology import convex_hull_image as chi 
#from skimage.util import invert
import skimage
import skimage.io 
from skimage import color
from skimage import feature
import matplotlib.pyplot as mpl
import skimage.filters

mpl.figure(figsize = (50,50))
cityScape = skimage.io.imread('SampleImage.jpg')


# Convert image to grayscale
cityScape = color.rgb2gray(cityScape)
mpl.imshow(cityScape,cmap = 'gray')
mpl.savefig('Grayscale.png')

#1. Roberts Edge Detection:
robertsEdge = skimage.filters.roberts(cityScape)
mpl.imshow(robertsEdge,cmap = 'gray')
mpl.savefig('Roberts Detection.png')


#2. Sobels Method: Better when edges are not vertical or horizontal; its more sensitive to 
#   diagonal edges
sobelEdge = skimage.filters.sobel(cityScape)
mpl.imshow(sobelEdge,cmap = 'gray')
mpl.savefig('Sobels Detection.png')

#3.Canny Edge
#Low sigma means noise will also be detected, more sensitive
cannyEdge = feature.canny(cityScape,sigma = 0.2)
mpl.imshow(cannyEdge,cmap = 'gray')
mpl.savefig('Canny Detection_Sigma_0.2.png')

# Best is to find an appropriate sigma value which detects the 
# edges we require
cannyEdge = feature.canny(cityScape,sigma = 1.7)
mpl.imshow(cannyEdge,cmap = 'gray')
mpl.savefig('Canny Detection_Sigma_1.7.png')

# As we increase sigma, heavier edges only are detected
cannyEdge = feature.canny(cityScape,sigma = 3)
mpl.imshow(cannyEdge,cmap = 'gray')
mpl.savefig('Canny Detection_Sigma_3.png')



