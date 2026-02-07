import rasterio
import numpy as np

raster_path = 'waterpath_data_service/data/kampala_dataset/scenarios/SSP1_2025/isoraster.tif'

with rasterio.open(raster_path) as src:
    data = src.read(1)
    nodata = src.nodata
    
    print(f"Raster shape: {data.shape}")
    print(f"Total pixels: {data.size}")
    print(f"NoData value: {nodata}")
    
    # Filter valid data properly
    valid_mask = data < 1e10
    valid_mask &= np.isfinite(data)
    
    valid_data = data[valid_mask]
    nodata_pixels = data[~valid_mask]
    
    print(f"\nValid pixels: {len(valid_data)}")
    print(f"NoData pixels: {len(nodata_pixels)}")
    
    if len(valid_data) > 0:
        print(f"\nValid data statistics:")
        print(f"  Min: {valid_data.min():.2f}")
        print(f"  Max: {valid_data.max():.2f}")
        print(f"  Mean: {valid_data.mean():.2f}")
        print(f"  Median: {np.median(valid_data):.2f}")
        print(f"  Sum: {valid_data.sum():.0f}")
        print(f"\nValue distribution:")
        print(f"  < 10,000: {(valid_data < 10000).sum()}")
        print(f"  10k-50k: {((valid_data >= 10000) & (valid_data < 50000)).sum()}")
        print(f"  50k-100k: {((valid_data >= 50000) & (valid_data < 100000)).sum()}")
        print(f"  100k-500k: {((valid_data >= 100000) & (valid_data < 500000)).sum()}")
        print(f"  > 500k: {(valid_data >= 500000).sum()}")
        
        # Show some sample values
        print(f"\nFirst 20 valid values: {valid_data[:20]}")
        print(f"Last 20 valid values: {valid_data[-20:]}")
