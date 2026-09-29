"""Explicit local extraction lineage, never treated as independently authenticated."""
from __future__ import annotations
import base64
import json
from typing import Literal

from pydantic import Field

from .metadata import Record, Tile, HEADER_LIMIT, digest_shape, safe_name
from .pds3_parser import parse_pds3
from .pds4_raster import parse_pds4_raster


class Manifest(Record):
    schema_version: Literal[1]
    original_format: Literal['PDS3', 'PDS4']
    original_name: str = Field(max_length=255)
    original_size: int = Field(gt=0, le=2**63-1)
    original_sha256: str
    original_label_base64: str = Field(max_length=1400000)
    tile_sha256: str
    tile: Tile
    extractor: Literal['lunargrid-tile-v1']


def apply_manifest(content, container, digest, mode, max_pixels):
    if len(content) > 2 * HEADER_LIMIT:
        raise ValueError('Tile provenance manifest exceeds 2 MiB')
    manifest = Manifest.model_validate_json(content)
    safe_name(manifest.original_name)
    digest_shape(manifest.original_sha256)
    if digest_shape(manifest.tile_sha256) != digest:
        raise ValueError('Tile provenance SHA-256 does not match uploaded TIFF')
    try:
        label = base64.b64decode(manifest.original_label_base64, validate=True)
    except ValueError as exc:
        raise ValueError('Invalid original label encoding in tile provenance') from exc
    if len(label) > HEADER_LIMIT:
        raise ValueError('Original label exceeds 1 MiB')
    original = (parse_pds3 if manifest.original_format == 'PDS3' else parse_pds4_raster)(
        label, manifest.original_name, manifest.original_size)
    manifest.tile.validate_for(original.layout, max_pixels)
    actual = container.layout
    if (actual.width, actual.height, actual.bands, actual.dtype) != (manifest.tile.width, manifest.tile.height, 1, original.layout.dtype):
        # Endianness of a TIFF container may differ; the physical numeric type must agree.
        import numpy as np
        if ((actual.width, actual.height, actual.bands) != (manifest.tile.width, manifest.tile.height, 1)
                or np.dtype(actual.dtype).name != np.dtype(original.layout.dtype).name):
            raise ValueError('Tile provenance dimensions/type conflict with TIFF')
    if mode not in {'auto', 'none', manifest.original_format.lower()}:
        raise ValueError('Forced metadata mode conflicts with original label in tile provenance')
    result = container if mode == 'none' else original.model_copy(deep=True)
    result.layout = actual.model_copy(update={key: getattr(original.layout, key) for key in ('scale', 'value_offset', 'special_constants', 'valid_min', 'valid_max')})
    result.file_name, result.file_size, result.sha256 = container.file_name, container.file_size, digest
    result.grid, result.geometry_status = container.grid, container.geometry_status
    result.ingestion_route = 'local_tile'
    if mode != 'none':
        result.label_location = 'original_attached_in_manifest' if manifest.original_format == 'PDS3' else 'original_xml_in_manifest'
    result.lineage = {'original_format': manifest.original_format, 'original_file_name': manifest.original_name,
        'original_product_id': original.product_id, 'original_sha256': manifest.original_sha256,
        'original_file_size': manifest.original_size, 'original_layout': original.layout.model_dump(),
        'original_label_sha256': original.label_sha256,
        'full_image_origin': [manifest.tile.x, manifest.tile.y], 'original_band': manifest.tile.band,
        'verification': 'tile SHA-256 and original label/layout validated; original full-file hash and extraction correspondence are local-tool claims',
        'extractor': manifest.extractor}
    result.warnings.append('Local extraction provenance: original file hash and tile correspondence are supplied claims; server verifies uploaded tile bytes and original label consistency, not original full-scene bytes.')
    return result.finish()
