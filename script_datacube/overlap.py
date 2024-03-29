import numpy as np
import xarray
import datacube
import gdal
from osgeo import osr
import csv
from PIL import Image


def get_lat_long(x, y, tif_file):
    xoffset, px_w, rot1, yoffset, rot2, px_h = tif_file.GetGeoTransform()
    posX = px_w * x + rot1 * y + xoffset
    posY = rot2 * x + px_h * y + yoffset

    # shift to the center of the pixel
    posX += px_w / 2.0
    posY += px_h / 2.0

    # get CRS from dataset
    crs = osr.SpatialReference()
    crs.ImportFromWkt(tif_file.GetProjectionRef())
    # create lat/long crs with WGS84 datum
    crsGeo = osr.SpatialReference()
    crsGeo.ImportFromEPSG(4326)  # 4326 is the EPSG id of lat/long crs
    t = osr.CoordinateTransformation(crs, crsGeo)
    (long, lat, z) = t.TransformPoint(posX, posY)

    return long, lat


whale_presence = gdal.Open('OpenDataCube_data/Whale_presence.tif')
mefts = gdal.Open('OpenDataCube_data/mEFTs-Global.tif')

band_whales = whale_presence.GetRasterBand(1)

stats = band_whales.GetStatistics(True, True)

print("Band Type={}".format(gdal.GetDataTypeName(band_whales.DataType)))

barray_whales = band_whales.ReadAsArray()
toshow = Image.fromarray(barray_whales)

check = np.where(barray_whales > 0.0)
print("Number of whales= "+str(len(check[0])))
mask_whales = np.ma.masked_where(barray_whales <= 0.0, barray_whales)
mask_whales = np.ma.filled(mask_whales, fill_value=0.0)

band_mefts = mefts.GetRasterBand(1)
barray_mefts = band_mefts.ReadAsArray()
mefts_plus_whale = barray_mefts * mask_whales

print(barray_mefts.min())
print(barray_mefts.max())
right_value_mefts = np.where(barray_mefts >= 0)

csv_mefts_plus_whales = []
csv_whales = []
for posY, posX in zip(check[0], check[1]):
    long, lat = get_lat_long(posX, posY, whale_presence)
    # print(mefts_plus_whale[posX][posY])
    csv_mefts_plus_whales.append([mefts_plus_whale[posY][posX], long, lat])
    csv_whales.append([barray_whales[posY][posX], long, lat])

csv_mefts = []
for posY in np.arange(0, barray_mefts.shape[0], 100):
    for posX in np.arange(0, barray_mefts.shape[1], 100):
        long, lat = get_lat_long(posX, posY, mefts)
        csv_mefts.append([barray_mefts[posY][posX], long, lat])

with open('mefts_and_whales.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(['value','long','lat'])
    wr.writerows(csv_mefts_plus_whales)

with open('whales.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(['value','long','lat'])
    wr.writerows(csv_whales)

with open('mefts.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(['value','long','lat'])
    wr.writerows(csv_mefts)




# print("Driver: {}/{}".format(whale_presence.GetDriver().ShortName,
#                              whale_presence.GetDriver().LongName))
# print("Size is {} x {} x {}".format(whale_presence.RasterXSize,
#                                     whale_presence.RasterYSize,
#                                     whale_presence.RasterCount))
# print("Projection is {}".format(whale_presence.GetProjection()))
# geotransform = whale_presence.GetGeoTransform()
# if geotransform:
#     print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
#     print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))


# print(band_array.GetNoDataValue())
# print(band_array.GetMinimum())
# print(band_array.GetMaximum())
# print(band_array.GetScale())
# print(band_array.GetUnitType())

# scanline = band.ReadRaster(xoff=0, yoff=0,
#                         xsize=band.XSize, ysize=1,
#                         buf_xsize=band.XSize, buf_ysize=1,
#                         buf_type=gdal.GDT_Float32)

# tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
# tuple_of_floats = np.array(tuple_of_floats)
# for line in range(0, band.YSize):
#     scanline = band.ReadRaster(xoff=0, yoff=0, xsize=band.XSize, ysize=1, buf_xsize=band.XSize, buf_ysize=1,
#                                buf_type=gdal.GDT_Float32)
