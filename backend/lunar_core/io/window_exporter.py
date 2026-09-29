"""Export tile-local measurements and optional, validated reference ground grid."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.transform import from_origin

from .geotiff_exporter import MOON_2000_WKT, CSV_COLUMNS


def write_bundle(output: Path, aligned, valid, result, params, ties, residuals, scores, ids):
    output.mkdir(exist_ok=True)
    src, ref = params.source_record, params.reference_record
    st, rt = params.source_tile, params.reference_tile
    grid = ref.grid.window(rt) if ref.grid else None
    options = dict(driver='GTiff', width=rt.width, height=rt.height, count=1,
                   dtype=aligned.dtype, compress='lzw', predictor=3)
    if grid:
        options.update(crs=MOON_2000_WKT, transform=from_origin(grid.west_lon, grid.north_lat, grid.pixel_size_deg, grid.pixel_size_deg))
    with warnings.catch_warnings(), rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(output / 'registered_output.tif', 'w', **options) as ds:
            ds.write(np.where(valid, aligned, 0), 1)
            ds.write_mask(valid.astype(np.uint8) * 255)
            ds.scales = (src.layout.scale,)
            ds.offsets = (src.layout.value_offset,)
            ds.update_tags(georeferenced=str(bool(grid)).lower(), source_product_id=src.product_id or '',
                reference_product_id=ref.product_id or '', radiometry='resampled raw DN; calibration in scale/offset',
                coordinate_convention='reference tile; zero-based pixel centers',
                independent_ground_validation='false', source_format=src.source_format, reference_format=ref.source_format)
        with rasterio.open(output / 'registered_output.tif') as ds:
            if bool(ds.crs) != bool(grid) or not np.array_equal(ds.dataset_mask() > 0, valid):
                raise ValueError('Export CRS/mask verification failed')
    columns = (*CSV_COLUMNS, 'src_full_x', 'src_full_y', 'ref_full_x', 'ref_full_y')
    src_origin = src.lineage.get('full_image_origin', [0, 0])
    ref_origin = ref.lineage.get('full_image_origin', [0, 0])
    with (output / 'tiepoints.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for identifier, tie, residual, score in zip(ids, ties, residuals, scores):
            sx, sy, rx, ry = map(float, tie)
            lat, lon = '', ''
            if grid:
                lat = grid.north_lat - (ry + .5) * grid.pixel_size_deg
                lon = ((grid.west_lon + (rx + .5) * grid.pixel_size_deg + 180) % 360) - 180
            writer.writerow([identifier, sx, sy, rx, ry, lat, lon, float(residual), float(score),
                sx + st.x + src_origin[0], sy + st.y + src_origin[1], rx + rt.x + ref_origin[0], ry + rt.y + ref_origin[1]])
    result['export'] = {'georeferenced': bool(grid), 'reference_tile_grid': grid.model_dump() if grid else None,
        'coordinate_convention': 'src_x/src_y/ref_x/ref_y: tile-local zero-based pixel centers; *_full_* include selected tile and extraction origins',
        'ground_coordinates': 'derived from validated reference grid; not independent ground truth' if grid else 'unavailable; CSV ground fields empty',
        'radiometry': 'float64 resampled raw DN; mask excludes invalid bilinear support and reference nulls; TIFF scales/offsets preserve calibration',
        'independent_ground_validation': False, 'valid_pixels': int(valid.sum())}
    write_dossier(output, result)


def write_dossier(output, result):
    output.mkdir(exist_ok=True)
    (output / 'registration_dossier.json').write_text(json.dumps(
        {'schema_version': 2, **result}, allow_nan=False, indent=2), encoding='utf-8')
