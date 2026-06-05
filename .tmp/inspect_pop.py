import rasterio, numpy as np
from pathlib import Path

bases = [
    'waterpath_data_service/data/kampala_level_4/baseline/human_emissions',
    'waterpath_data_service/data/test/baseline/human_emissions',
    'waterpath_data_service/data/test_dhaka/baseline/human_emissions',
    'waterpath_data_service/data/uganda/baseline/human_emissions',
]
for base in bases:
    print('==', base)
    arrs = {}
    for name in ['pop_urban.tif', 'pop_rural.tif']:
        p = Path(base) / name
        with rasterio.open(p) as src:
            a = src.read(1).astype('float64'); nd = src.nodata
            mask = np.isfinite(a)
            if nd is not None:
                mask &= ~np.isclose(a, nd)
            arrs[name] = (a, mask)
            v = a[mask]
            print(f'  {name:14s} sum={v.sum():.1f} min={v.min():.4g} max={v.max():.4g} n_nonzero={int((v!=0).sum())} n_valid={int(mask.sum())}')
    au, mu = arrs['pop_urban.tif']; ar, mr = arrs['pop_rural.tif']
    both = mu & mr
    print(f'  identical={np.array_equal(au,ar)} sum_u+r(valid both)={(au[both]+ar[both]).sum():.1f}')
