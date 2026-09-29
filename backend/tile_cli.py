"""Local bounded raw PDS extraction. No server filesystem path ingestion endpoint."""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import warnings

import numpy as np
import rasterio

from .lunar_core.io.ingestion import inspect_image, read_tile, sha256_file, header
from .lunar_core.io.metadata import Tile, DEFAULT_TILE_PIXELS, HEADER_LIMIT
from .lunar_core.io.pds3_parser import parse_odl


def extract(image: Path, output: Path, tile: Tile, xml: Path | None = None):
    if image.is_symlink() or (xml and xml.is_symlink()):
        raise ValueError('Use regular local files, not symlinks')
    label = None
    if xml:
        with xml.open('rb') as stream:
            label = stream.read(HEADER_LIMIT + 1)
    record = inspect_image(image, image.name, xml=label)
    if record.source_format == 'IMAGE_ONLY' or record.layout.decoder != 'raw':
        raise ValueError('This extractor accepts validated raw PDS3/PDS4 rasters')
    tile.validate_for(record.layout, DEFAULT_TILE_PIXELS)
    decoded = read_tile(image, record, tile)
    if not decoded.valid.any():
        raise ValueError('Selected tile is all-null; choose another window')
    manifest_path = output.with_suffix('.provenance.json')
    if output.exists() or manifest_path.exists():
        raise ValueError('Output exists; choose a new tile filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings(), rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(output, 'w', driver='GTiff', width=tile.width, height=tile.height,
                count=1, dtype=decoded.raw.dtype, compress='lzw', tiled=True) as ds:
                ds.write(decoded.raw, 1)
                ds.write_mask(decoded.valid.astype(np.uint8)*255)
                ds.scales, ds.offsets = (record.layout.scale,), (record.layout.value_offset,)
        if record.source_format == 'PDS3':
            label = header(image)
            _, end = parse_odl(label)
            label = label[:end]
        manifest = {'schema_version':1, 'original_format':record.source_format, 'original_name':image.name,
            'original_size':record.file_size, 'original_sha256':record.sha256,
            'original_label_base64':base64.b64encode(label).decode('ascii'),
            'tile_sha256':sha256_file(output), 'tile':tile.model_dump(), 'extractor':'lunargrid-tile-v1'}
        manifest_path.write_text(json.dumps(manifest, indent=2),encoding='utf-8')
    except BaseException:
        output.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return {'tile':str(output), 'provenance':str(manifest_path), 'origin':[tile.x,tile.y],
            'bytes_read_for_pixels':decoded.bytes_read, 'original_sha256':record.sha256,
            'note':'Original SHA-256 uses a streaming full-file pass; raster decoding reads only the selected window.'}


def main():
    parser=argparse.ArgumentParser(description='Extract <=1,048,576 pixels locally; upload TIFF and its .provenance.json on the same image card. Full-scene geometry is not inferred.')
    parser.add_argument('image', type=Path)
    parser.add_argument('--xml', type=Path)
    parser.add_argument('--x', type=int, required=True); parser.add_argument('--y',type=int,required=True)
    parser.add_argument('--width',type=int,default=512); parser.add_argument('--height',type=int,default=512)
    parser.add_argument('--band',type=int,default=0); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(extract(args.image,args.output,Tile(x=args.x,y=args.y,width=args.width,height=args.height,band=args.band),args.xml),indent=2))


if __name__ == '__main__':
    main()
