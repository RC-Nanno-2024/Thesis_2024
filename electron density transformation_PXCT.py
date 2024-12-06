
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from scipy.stats import norm
import seaborn as sns
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
from lmfit import Model


# Conversion formula
# im_delta_from_tiff = im_tiff*(high_cutoff-low_cutoff)/(2^16-1) + low_cutoff;
# im_edensity_from_tiff = im_delta_from_tiff*factor_edensity;
#factor = 8.845490e-04
#pixel size = 2.788431e-08
low_cutoff = -2.574043e-06
high_cutoff = 1.032500e-05
factor_edensity = 9.283780e+04
image = plt.imread('image in.tif')
im_delta_from_tiff = (image*(high_cutoff-low_cutoff)/(2**16-1)) + low_cutoff;
im_edensity_from_tiff = im_delta_from_tiff*9.283780e+04
plt.imshow(im_edensity_from_tiff)
plt.colorbar(orientation="vertical") 
im_edensity_from_tiff_sample_1=np.where((im_edensity_from_tiff>=0.3) & (im_edensity_from_tiff<=0.5), im_edensity_from_tiff, 0)
plt.imshow(im_edensity_from_tiff_sample_1)
plt.colorbar(orientation="vertical")   

plt.hist(im_edensity_from_tiff, bins=200, ec ='k', label = 'Histogram')
plt.legend()
print(im_edensity_from_tiff)
plt.imshow(im_edensity_from_tiff)


#Plotting the histogram and the density curve from the data
plt.hist(im_edensity_from_tiff, bins=500, ec='k', label = "Histogram")
sns.distplot(im_edensity_from_tiff, color= 'Green', label = 'Density Plot')
plt.legend()


#Calculation for the values coming from air
im_edensity_from_tiff_air=im_edensity_from_tiff[(im_edensity_from_tiff >= -0.05) 
                        & (im_edensity_from_tiff <=0.05)]
plt.hist(im_edensity_from_tiff_air, bins=200, ec ='k', label = 'Histogram')
plt.legend()
mu_air, sigma_air = norm.fit(im_edensity_from_tiff_air)
print(mu_air, sigma_air)


#Removing the values of edensity coming from air
im_edensity_from_tiff_sample = im_edensity_from_tiff[(im_edensity_from_tiff >= 0.6)]
im_edensity_from_tiff_sample = [i+0.0027424001251622797 for i in 
   im_edensity_from_tiff_sample] #Adding the amount (mean_air) to shift the values to the right
print(im_edensity_from_tiff_sample)


#Segmenting data set near 0.4 e-density
im_edensity_from_tiff_new=im_edensity_from_tiff[(im_edensity_from_tiff >= 0.325) 
                        & (im_edensity_from_tiff <=0.425)]
im_edensity_from_tiff_new = [i+0.0027424001251622797 for i in 
   im_edensity_from_tiff_new]
plt.hist(im_edensity_from_tiff_new, bins=200, ec ='k', label = 'Histogram')
plt.legend()
mu_new, sigma_new = norm.fit(im_edensity_from_tiff_new)
print(mu_new, sigma_new)


