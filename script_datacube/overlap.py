import numpy as np
import xarray
import datacube
import gdal
import struct

whale_presence = gdal.Open('OpenDataCube_data/Whale_presence.tif')
mefts = gdal.Open('OpenDataCube_data/mEFTs-Global.tif')

print("Driver: {}/{}".format(whale_presence.GetDriver().ShortName,
                             whale_presence.GetDriver().LongName))
print("Size is {} x {} x {}".format(whale_presence.RasterXSize,
                                    whale_presence.RasterYSize,
                                    whale_presence.RasterCount))
print("Projection is {}".format(whale_presence.GetProjection()))
geotransform = whale_presence.GetGeoTransform()
if geotransform:
    print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
    print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

band_whales = whale_presence.GetRasterBand(1)

stats = band_whales.GetStatistics(True, True)

print("Band Type={}".format(gdal.GetDataTypeName(band_whales.DataType)))
barray_whales = band_whales.ReadAsArray()

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
check = np.where(barray_whales>0.0)
print(len(check[0]))
mask_whales = np.ma.masked_where(barray_whales <= 0.0, barray_whales)
print(len(mask_whales[0]))

band_mefts = mefts.GetRasterBand(1)
barray_mefts = band_whales.ReadAsArray()
final_mask = np.ma.masked_where(barray_whales <= 0.0, barray_mefts)
# masked = np.ma.
