import rasterio
import numpy as np
import sys

raster_path = sys.argv[1] if len(sys.argv) > 1 else 'waterpath_data_service/data/kampala_dataset/scenarios/SSP1_2025/isoraster.tif'

with rasterio.open(raster_path) as src:
    data = src.read(1)
    print(f'Shape: {data.shape}')
    print(f'NoData: {src.nodata}')
    print(f'CRS: {src.crs}')
    print(f'Min: {np.nanmin(data)}')
    print(f'Max: {np.nanmax(data)}')
    print(f'Mean: {np.nanmean(data)}')
    print(f'Non-zero pixels: {np.count_nonzero(data)}')
    print(f'Total sum: {np.nansum(data)}')
    
    # Check for positive values
    positive = data[data > 0]
    if len(positive) > 0:
        print(f'Positive values count: {len(positive)}')
        print(f'Sample positive values: {positive[:10]}')
    else:
        print('No positive values found')
    
    # Check unique values
    unique = np.unique(data)
    print(f'Unique values (first 20): {unique[:20]}')
