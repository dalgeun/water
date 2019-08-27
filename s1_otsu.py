
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
  arr2[arr2<=0] = np.nan
  arr2[arr2>1] = np.nan

  # arr3 = arr2
  arr3 = np.log10(arr2)
  arr3_nan_removed = arr3[~np.isnan(arr3)]    
  # plt.imshow(arr3)
  BINS = np.linspace(np.amin(arr3_nan_removed), np.amax(arr3_nan_removed), 1000)
  # BINS = np.arange(np.amin(arr3_nan_removed), np.amax(arr3_nan_removed), 0.001)
  # hist, bins =np.histogram(arr2.ravel(), BINS)  
  hist, bins =np.histogram(arr3_nan_removed.ravel(), BINS)
  hist = hist.astype(float)
  bins = bins[:-1]
  plt.bar(bins, hist, width = 0.0008, align='center')
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
  return threshold, arr3


path_to_img_data = "D:\dg_work\water\img_mask\subset_water.tif"
thres, img = thres_otsu(path_to_img_data)
print("Otsu threshold = ", thres)


img_n = img[~np.isnan(img)] 
img_min = np.amin(img_n)
img_max = np.amax(img_n)
BINS = np.linspace(img_min, img_max, 1000)
hist_i, bins_i =np.histogram(img_n.ravel())
hist_i = hist_i.astype(float)
bins_i = bins_i[:-1]
plt.bar(bins_i, hist_i, width = 0.1, align = 'center')
plt.imshow(img)

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




#test
def hist_img(img):
  # img_nan = img[~np.isnan(img)]
  img_eq_nan = img_eq
  # BINS_eq = np.arange(np.amin(img_eq_nan), np.amax(img_eq_nan), 0.01)
  BINS_eq = np.linspace(np.amin(img_nan), np.amax(img_nan), 1024)
  hist_eq, bins_eq =np.histogram(img_nan.ravel(), BINS_eq)
  hist_eq = hist_eq.astype(float)
  bins_eq = bins_eq[:-1]
  plt.bar(bins_eq, hist_eq, width = 0.01, align = 'center')
  return hist_eq, bins_eq

def otsu_img(hist_eq, bins_eq):
  #### OTSU(equalization) strat of image equalization
  weight1 = np.cumsum(hist_eq)
  weight2 = np.cumsum(hist_eq[::-1])[::-1]
  # class means for all possible thresholds
  mean1 = np.cumsum(hist_eq * bins_eq) / weight1
  mean2 = (np.cumsum((hist_eq * bins_eq)[::-1]) / weight2[::-1])[::-1]
  # Clip ends to align class 1 and class 2 variables:
  # The last value of `weight1`/`mean1` should pair with zero values in
  # `weight2`/`mean2`, which do not exist.
  variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
  idx = np.nanargmax(variance12)
  threshold = bins_eq[:-1][idx]
  print("OTSU threshold = ", threshold)
  return threshold

def img_equal(arr3):
  arr3_nan_removed = arr3[~np.isnan(arr3)]
  # arr3_nan_removed = np.ma.masked_equal(arr3, np.isnan(arr3))
  # arr3_nan_removed = np.ma.masked_equal(arr3, 0)
  # arr3_nan_removed = arr3
  # arr3_nan_removed = np.ma.masked_where(np.isnan(arr3), arr3)
  # BINS = np.arange(np.amin(arr3_nan_removed), np.amax(arr3_nan_removed), 0.01)
  BINS = np.linspace(np.amin(arr3_nan_removed), np.amax(arr3_nan_removed), 1024)
  hist, bins =np.histogram(arr3_nan_removed.ravel(), BINS)
  hist = hist.astype(float)
  bins = bins[:-1]
  plt.bar(bins, hist, width = 0.8, align='center')

  cdf = hist.cumsum()
  # cdf_normalized = cdf * hist.max() / cdf.max()
  # plt.plot(cdf_normalized)
  cdf_m = np.ma.masked_equal(cdf, 0)  
  # cdf_m = (cdf_m - cdf_m.min())*len(bins)/(cdf_m.max()-cdf_m.min())
  cdf_m = ((cdf_m - cdf_m.min())*-1/(cdf_m.max()-cdf_m.min()))+1
  cdf = np.ma.filled(cdf_m, 0).astype('uint8')
  print('cdf max is = ', cdf_m.max())
  plt.figure()
  plt.plot(cdf_m)
  img_eq = np.interp(arr3.ravel(), bins, cdf_m)
  img_eq = img_eq.reshape(arr3.shape)
  plt.figure()
  plt.imshow(img_eq)

  return img_eq

def otsu_water(img_eq, threshold):
  binary_eq = img_eq <= threshold
  otsu_img_eq = img_eq
  otsu_img_eq[~binary_eq] = np.nan
  otsu_img_eq[binary_eq] = 1
  plt.figure()
  plt.imshow(otsu_img_eq)
  plt.figure()
  plt.imshow(binary_eq)
  ## compute water area
  water_pixels = np.count_nonzero(otsu_img_eq == 1)
  water_area = water_pixels * 0.0001
  print("water area is ", water_area, "km^2")
  return water_area


ds = gdal.Open(path_to_img_data)
  # arys =[]
band = ds.GetRasterBand(1)
arr = band.ReadAsArray()

arr2 = arr
arr2[arr2<0] = np.nan
arr2[arr2>1] = np.nan

np.amax(arr2)
# arr3= np.ma.masked_invalid(arr2)

arr2 = np.where(arr2 <0, np.nan, arr2)
arr2 = np.where(arr2 >1, np.nan, arr2)
arr2 = np.ma.array(arr2, mask=np.isnan(arr2))
# arr3 = np.ma.masked_equal(arr2, np.isnan(arr2))

img_eq = img_equal(arr2)

hist_eq, bins_eq = hist_img(arr2)
threshold = otsu_img(hist_eq, bins_eq)
water_area = otsu_water(arr2, threshold)