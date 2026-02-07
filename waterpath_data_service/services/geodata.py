import pygadm, pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import os
import rasterio
import rasterio.features
import rasterio.mask
from rasterio.transform import from_origin
import numpy as np


def geonames(admin: str, level: int) -> list:

    areas = [x.strip(" ") for x in admin.split(",")]
    names = []
    for area in areas:
        name = pygadm.Names(admin=area, content_level=level)
        names.extend(name["NAME_" + str(level)].tolist())

    names_df = pd.DataFrame(columns=["gid", "name"])
    names_df["gid"] = areas
    names_df["name"] = names
    return names_df


def shapefile(areas: str, path: str) -> str:
    level = areas[0].count(".")
    gdf = pygadm.Items(admin=areas, content_level=level)
    geodata_path = "geodata"
    path = str(Path(path))
    geo_path = path + os.sep + geodata_path
    if not os.path.isdir(geo_path):
        os.mkdir(geo_path)
    file_path = geo_path + os.sep +"geodata.shp"
    file = gdf.to_file(file_path)
    return file_path


def geofilter(data: str, path: str) -> str:
    geodata_path = str(Path(path)) + os.sep + "geodata"
    shapefile_path = geodata_path + os.sep + "geodata.shp"
    polygon_gdf = gpd.read_file(shapefile_path)
    df = pd.read_json(data)
    geometry = [Point(xy) for xy in zip(df["lon"], df["lat"])]
    points_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=polygon_gdf.crs)
    points_within = points_gdf[points_gdf.within(polygon_gdf.unary_union)]
    points_within = points_within.drop(columns=["alpha3", "geometry"])
    return points_within

def resample_raster(path: str):
    
    # Step 1: Load vector data
    vect_gadm = gpd.read_file(str(Path(path)) + os.sep + "geodata" + os.sep + "geodata.shp")

    # Step 2: Add unique IDs to features
    vect_gadm["iso"] = range(1, len(vect_gadm) + 1)

    # Step 3: Define raster domain (0.5 degree resolution)
    # Assuming bounds cover the vector layer
    minx, miny, maxx, maxy = vect_gadm.total_bounds
    
    res = 0.5
    width = int((maxx - minx) / res)
    height = int((maxy - miny) / res)
    
    transform = from_origin(minx, maxy, res, res)
    
    # Step 4: Rasterize vector layer
    rast_iso_array = rasterio.features.rasterize(
        ((geom, value) for geom, value in zip(vect_gadm.geometry, vect_gadm["iso"])),
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype='int64'
    )

    # Save isoraster
    isoraster_path = str(Path(path))+ os.sep + "geodata" + os.sep + "isoraster.tif"
    os.makedirs(os.path.dirname(isoraster_path), exist_ok=True)

    with rasterio.open(
        isoraster_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=rast_iso_array.dtype,
        crs=vect_gadm.crs,
        transform=transform,
    ) as dst:
        dst.write(rast_iso_array, 1)
    pop_urban = str(Path(__file__).parent.parent) + os.sep + "static" + os.sep + "data" + os.sep + "pop_urban.tif"
    pop_rural = str(Path(__file__).parent.parent) + os.sep + "static" + os.sep + "data" + os.sep + "pop_rural.tif"
    crop_and_mask_by_iso(pop_urban, rast_iso_array, transform, vect_gadm.crs, str(Path(path)) + os.sep + "geodata" + os.sep + "pop_urban.tif")
    

# Helper: Crop and mask population raster using non-zero area of iso raster
def crop_and_mask_by_iso(pop_path, iso_array, transform, crs, output_path):
    with rasterio.open(pop_path) as src:
        pop_data = src.read(1)
        pop_transform = src.transform
        print(pop_path)
        # Reproject iso array to match pop raster if needed
        if transform != pop_transform:
            raise ValueError("Transform mismatch – reprojection needed")

        mask_geometry = []
        for shape, value in rasterio.features.shapes(iso_array, transform=transform):
            if value > 0:
                mask_geometry.append(shape)

        out_image, out_transform = rasterio.mask.mask(src, mask_geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)