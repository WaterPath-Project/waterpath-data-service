import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from waterpath_data_service.services.projections import calculate_zonal_population
import pandas as pd

# Test the zonal calculation
raster_path = Path('waterpath_data_service/data/kampala_dataset/scenarios/SSP1_2025/isoraster.tif')
shapefile_path = Path('waterpath_data_service/data/kampala_dataset/default/geodata/geodata.shp')
csv_path = Path('waterpath_data_service/data/kampala_dataset/default/human_emissions.csv')

# Determine the correct GID column
df = pd.read_csv(csv_path)
sample_gid = str(df['gid'].iloc[0])
dot_count = sample_gid.count(".")
shapefile_gid_column = f"GID_{dot_count}"

print(f"Sample CSV gid: {sample_gid}")
print(f"Dot count: {dot_count}")
print(f"Using shapefile column: {shapefile_gid_column}")

print("\nTesting zonal population calculation...")
try:
    results = calculate_zonal_population(raster_path, shapefile_path, gid_column=shapefile_gid_column)
    print(f"\nNumber of results: {len(results)}")
    print(f"First 5 results:")
    for i, (gid, pop) in enumerate(list(results.items())[:5]):
        print(f"  {gid}: {pop:.2f}")
    
    total = sum(results.values())
    print(f"\nTotal population: {total:.0f}")
    
    # Check for zeros
    zeros = [gid for gid, pop in results.items() if pop == 0]
    print(f"Number of areas with 0 population: {len(zeros)}")
    if len(zeros) > 0 and len(zeros) < 10:
        print(f"Zero population areas: {zeros}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
