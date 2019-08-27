
import os
import numpy as np
import gdal
import matplotlib.pyplot as plt  


def thres_otsu(path_to_img_data):

  # ds = gdal.Open(path_to_img_data,  gdal.GA_ReadOnly)
  ds = gdal.Open(path_to_img_data)
  # arys =[]
  band = ds.GetRasterBand(1)
  arr = band.ReadAsArray()
  [cols, rows] = arr.shape
  arr_min = np.amin(arr)
  arr_max = np.amax(arr) 

  arr2 = arr
  arr2[arr2<=0] = float('nan')
  arr2[arr2>=1] = float('nan')

  arr2_log = np.log10(arr2)
  arr2_log_nan_removed = arr2_log[~np.isnan(arr2_log)]  
  # plt.imshow(arr3)
  # BINS = np.arange(0, 1, 0.0001)  
  BINS = np.arange(np.amin(arr2_log_nan_removed), np.amax(arr2_log_nan_removed), 0.01)
  # hist, bins =np.histogram(arr2.ravel(), BINS)  
  hist, bins =np.histogram(arr2_log_nan_removed.ravel(), BINS)
  hist = hist.astype(float)
  bins = bins[:-1]
  plt.bar(bins, hist, width = 0.008, align='center')
  # total_counts = np.sum(hist)
  
  weight1 = np.cumsum(hist)
  weight2 = np.cumsum(hist[::-1])[::-1]
  # class means for all possible thresholds
  mean1 = np.cumsum(hist * bins) / weight1
  mean2 = (np.cumsum((hist * bins)[::-1]) / weight2[::-1])[::-1]
  # Clip ends to align class 1 and class 2 variables:
  # # The last value of `weight1`/`mean1` should pair with zero values in
  # # `weight2`/`mean2`, which do not exist.
  variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
  idx = np.nanargmax(variance12)
  threshold = bins[:-1][idx]
  '''
  Examples
  >> from skimage.data import camera
  >> image = camera
  >> thresh =thres_otsu(image)
  >> binary = image <= thres
  '''
  return threshold, arr2_log


path_to_img_data = "D:\dg_work\water\img_mask\subset_water.tif"
thres, img = thres_otsu(path_to_img_data)
print("Otsu threshold = ", thres)

""" 
img_n = img[~np.isnan(img)] 
img_min = np.amin(img_n)
img_max = np.amax(img_n)
hist, bins =np.histogram(img.ravel(), np.arange(img_min, img_max, 0.001))
hist = hist.astype(float)
bins = bins[:-1]
plt.bar(bins, hist, width = 0.0008, align = 'center')
 """

# apply otsu threshold
binary = img <= thres
otsu_img = img
otsu_img[~binary] = np.nan
otsu_img[binary] = 1

plt.imshow(img)
plt.imshow(otsu_img)
plt.imshow(binary)

## compute water area
water_pixels = np.count_nonzero(otsu_img == 1)
water_area = water_pixels * 0.0001
print("water area is ", water_area, "km^2")

## ouput image
outfile = "d:\\dg_work\\water\\water_area_bi_otsu_log.tif"
ds = gdal.Open(path_to_img_data)
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()
[cols, rows] = arr.shape

driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(otsu_img)
# outdata.GetRasterBand(1).SetNoDataValue()##if you want these values transparent
outdata.FlushCache() ##saves to disk!!
outdata = None
band=None
ds=None




from skimage import filters
from skimage import io

ds = gdal.Open(path_to_img_data)
  # arys =[]
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()
[cols, rows] = arr.shape
arr_min = np.amin(arr)
arr_max = np.amax(arr)

arr2 = arr
arr2[arr2<=0] = float('nan')
arr2[arr2>=1] = float('nan')

arr3 = np.log10(arr2)

arr3_nan_removed = arr3[~np.isnan(arr3)]  
BINS = np.arange(np.amin(arr3_nan_removed), np.amax(arr3_nan_removed), 0.01)
hist, bins =np.histogram(arr3_nan_removed.ravel(), BINS)
hist = hist.astype(float)
bins = bins[:-1]
plt.bar(bins, hist, width = 0.008, align='center')

cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max() / cdf.max()
cdf_m = np.ma.masked_equal(cdf, float('nan'))
cdf_m = (cdf_m - cdf_m.min())*len(bins)/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, float('nan')).astype(float)
plt.plot(cdf)
plt.plot(cdf_m)

img_eq = np.interp(arr3, bins, cdf_m)

plt.imshow(img_eq)
img_eq_nan = img_eq[~np.isnan(img_eq)]
BINS_eq = np.arange(np.amin(img_eq_nan), np.amax(img_eq_nan), 0.1)
hist_eq, bins_eq =np.histogram(img_eq.ravel(), BINS_eq)
hist_eq = hist_eq.astype(float)
bins_eq = bins_eq[:-1]
plt.bar(bins_eq, hist_eq, width = 0.05, align = 'center')

weight1 = np.cumsum(hist_eq)
weight2 = np.cumsum(hist_eq[::-1])[::-1]
# class means for all possible thresholds
mean1 = np.cumsum(hist_eq * bins_eq) / weight1
mean2 = (np.cumsum((hist_eq * bins_eq)[::-1]) / weight2[::-1])[::-1]
# Clip ends to align class 1 and class 2 variables:
# # The last value of `weight1`/`mean1` should pair with zero values in
# # `weight2`/`mean2`, which do not exist.
variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
idx = np.nanargmax(variance12)
threshold = bins_eq[:-1][idx]


# apply otsu threshold
binary_eq = img_eq <= threshold
otsu_img_eq = img_eq
otsu_img_eq[~binary_eq] = np.nan
otsu_img_eq[binary_eq] = 1

plt.imshow(img_eq)
plt.imshow(otsu_img_eq)
plt.imshow(binary_eq)

## compute water area
water_pixels = np.count_nonzero(otsu_img == 1)
water_area = water_pixels * 0.0001
print("water area is ", water_area, "km^2")