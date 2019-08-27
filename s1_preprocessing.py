##Full Script
import os

import snappy
from snappy import Product
from snappy import ProductIO
from snappy import ProductUtils
from snappy import WKTReader
from snappy import HashMap
from snappy import GPF
from snappy import jpy

# For shapefile
import shapefile #pip install pyshp
import pygeoif

 
"""
def plotBand(product, band, vmin, vmax):

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    band = product.getBand(band)
    w = band.getRasterWidth()
    h = band.getRasterHeight()
    print(w, h)

    band_data = np.zeros(w * h, np.float32)
    band.readPixels(0, 0, w, h, band_data)

    band_data.shape = h, w
    width = 12
    height = 12
    plt.figure(figszie=(width, height))
    imgplot = plt.imshow(band_data, cmap=plt.cm.binary, vmin=vmin, vmax=vmax)

    return imgplot
  """

def showProductInformation(product):
    width = product.getSceneRasterWidth()
    print("Width: {} px".format(width))
    height = product.getSceneRasterHeight()
    print("Height: {} px".format(height))
    name = product.getName()
    print("Name: {}".format(name))
    band_names = product.getBandNames()
    print("Band names: {}".format(", ".join(band_names)))

def shpToWKT(shp_path):
    r = shapefile.Reader(shp_path) #shp파일 경로
    g = []
    for s in r.shapes():
        g.append(pygeoif.geometry.as_shape(s))
    m = pygeoif.MultiPoint(g)
    return str(m.wkt).replace("MULTIPOINT", "POLYGON(") + ")" 

########### Image Pre-processing ###########
# orbit file application
def applyOrbit(product):
    parameters = HashMap()
    parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
    parameters.put('polyDegree', '3')
    parameters.put('continueOnFail', 'false')
    return GPF.createProduct('Apply-Orbit-File', parameters, product)

def subset(product, shpPath):
    parameters = HashMap()
    wkt = shpToWKT(shpPath)
    SubsetOp = jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
    geometry = WKTReader().read(wkt)
    parameters.put('copyMetadata', True)
    parameters.put('geoRegion', geometry)
    return GPF.createProduct('Subset', parameters, product)

def calibration(product):
    parameters = HashMap()
    parameters.put('outputSigmaBand', True)
    parameters.put('sourcebands', 'Intensity_VV')
    parameters.put('selectedPolarisations', "VV")
    parameters.put('outputImageScaleInDb', False)
    return GPF.createProduct("Calibration", parameters, product)

def speckleFilter(product):
    parameters = HashMap()

    filterSizeY = '5'
    filterSizeX = '5'

    parameters.put('sourceBands', 'Sigma0_VV')
    parameters.put('filter', 'Lee')
    parameters.put('filterSizeX', filterSizeX)
    parameters.put('filterSizeY', filterSizeY)
    parameters.put('dampaingFactor', '2')
    parameters.put('estimateENL', 'true')
    parameters.put('enl', '1.0')
    parameters.put('numLooksStr', '1')
    parameters.put('targetWindowSizeStr', '3x3')
    parameters.put('sigmaStr', '0.9')
    parameters.put('anSize', '50')
    return GPF.createProduct('Speckle-Filter', parameters, product)

def terrainCorrection(product):
    parameters = HashMap()
    parameters.put('demName', 'SRTM 3Sec')
    parameters.put('pixelSpacingInMeter', 10.0)
    parameters.put('sourceBands', 'Sigma0_VV')

    return GPF.createProduct("Terrain-Correction", parameters, product)

# Water Mask Processing

def generateBinaryFlood(product):
    parameters = HashMap()

    BandDescriptor = snappy.jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    targetBand = BandDescriptor()
    targetBand.name = 'flooded'
    targetBand.type = 'uint8'
    targetBand.expression = '(Sigma0_VV < 2.22E-2) ? 1: 0'
    targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
    targetBands[0] = targetBand
    parameters.put('targetBands', targetBands)

    return GPF.createProduct('BandMaths', parameters, product)

def maskKnownWater(product):
    # Add land cover band
    
    parameters = HashMap()
    parameters.put("landCoverNames", "GlobCover")
    mask_with_land_cover = GPF.createProduct('AddLandCover', parameters, product)
    del parameters

    # Create binary water band
    BandDescriptor = snappy.jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    parameters = HashMap()
    targetBand = BandDescriptor()
    targetBand.name = 'BinaryWater'
    targetBand.type = 'uint8'
    targetBand.expression = '(land_cover_GlobCover == 210) ? 0 : 1'
    targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
    targetBands[0] = targetBand
    parameters.put('targetBands', targetBands)
    water_mask = GPF.createProduct('BandMaths', parameters, mask_with_land_cover)
    del parameters
    

    parameters = HashMap()

    BandDescriptor = snappy.jpy.get_type('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor')
    try:
        water_mask.addBand(product.getBand("flooded"))
    except:
        pass
    
    targetBand = BandDescriptor()
    targetBand.name = 'Sigma0_VV_Flooded_Masked'
    targetBand.type = 'uint8'
    #targetBand.expression = '(BinaryWater == 1 && flooded == 1) ? 1 : 0'
    targetBand.expression = '(flooded == 1) ? 1 : 0'
    targetBands = snappy.jpy.array('org.esa.snap.core.gpf.common.BandMathsOp$BandDescriptor', 1)
    targetBands[0] = targetBand
    parameters.put('targetBands', targetBands)

    return GPF.createProduct('BandMaths', parameters, water_mask)


if __name__ == "__main__":
    ## GPF Initialization
    GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

    ## Prodcut initialization
    path_to_sentinel_data = "D:\dg_work\water\img\S1B_IW_GRDH_1SDV_20190503T213123_20190503T213148_016085_01E41C_1881.zip"
    product = ProductIO.readProduct(path_to_sentinel_data)
    showProductInformation(product)

    path_to_shp_data = "D:\dg_work\water\shp\soyang_shape\soyang_4326.shp"
    product_orbitfile = applyOrbit(product)
    product_subset = subset(product_orbitfile, path_to_shp_data)
    showProductInformation(product_subset)

    # Apply rematinder of processing steps in a nested function call
    product_preprocessed = terrainCorrection(
        speckleFilter(
            calibration(
                product_subset
            )
        )
    )
    path_to_preprocessed_data = "D:\\dg_work\\water\\img_preprocessed\\img_preprocessed"
    ProductIO.writeProduct(product_preprocessed, path_to_preprocessed_data, 'GeoTIFF')

    product_binaryflood = maskKnownWater(
        generateBinaryFlood(
            product_preprocessed
        )
    )

    path_to_masked_data = "D:\\dg_work\\water\\img_mask\\flooded_mask"
    ProductIO.writeProduct(product_binaryflood, path_to_masked_data, 'GeoTIFF')

    path_to_preprocessed_img = path_to_preprocessed_data +".tif"
    path_to_subset = "D:\dg_work\water\img_mask\subset_water.tif"
    ClipByPolygon = 'gdalwarp -overwrite -dstnodata 1 -srcnodata 0 -cutline %s -crop_to_cutline %s %s' \
        % (path_to_shp_data, path_to_preprocessed_img, path_to_subset)
    os.system(ClipByPolygon)

    del ClipByPolygon

    path_to_masked_img = path_to_masked_data + ".tif"
    path_to_subset_water_bi = "D:\dg_work\water\img_mask\subset_water_bi.tif"
    ClipByPolygon = 'gdalwarp -overwrite -cutline %s -crop_to_cutline %s %s' \
        % (path_to_shp_data, path_to_masked_img, path_to_subset_water_bi)
    os.system(ClipByPolygon)