"""Shared, resource-bounded image inspection and window decoding."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import rasterio
from rasterio.windows import Window
from pyproj import CRS

from .metadata import DEFAULT_TILE_PIXELS, HEADER_LIMIT, ImageMetadata, LunarGrid, RasterLayout, Tile, safe_name
from .pds3_parser import detect_pds3, parse_pds3
from .pds4_raster import parse_pds4_raster
from .geotiff_exporter import MOON_CRS
from .tile_manifest import apply_manifest


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def header(path):
    with Path(path).open('rb') as stream:
        return stream.read(HEADER_LIMIT)


def container_metadata(path: Path, name: str, max_pixels=16_777_216):
    with path.open('rb') as stream:
        magic = stream.read(4)
    if magic in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(path, driver='GTiff') as ds:
                if not 1 <= ds.count <= 4 or len(set(ds.dtypes)) != 1:
                    raise ValueError('TIFF requires 1..4 bands of one allowlisted physical type')
                if max(h * w for h, w in ds.block_shapes) > max_pixels:
                    raise ValueError('TIFF block exceeds decoder budget; locally re-tile it first')
                dtype = np.dtype(ds.dtypes[0]).str
                layout = RasterLayout(width=ds.width, height=ds.height, bands=ds.count,
                    dtype=dtype, sample_type=ds.dtypes[0], decoder='container', container_format='TIFF', scale=ds.scales[0], value_offset=ds.offsets[0])
                grid = None
                notes = []
                if ds.crs is not None:
                    transform = ds.transform
                    if CRS.from_wkt(ds.crs.to_wkt()).equals(MOON_CRS, ignore_axis_order=True) and transform.b == transform.d == 0 and transform.a > 0 and transform.e == -transform.a:
                        grid = LunarGrid(west_lon=transform.c, north_lat=transform.f, pixel_size_deg=transform.a)
                        if transform.f - ds.height * transform.a < -90 or ds.width * transform.a > 360:
                            raise ValueError('TIFF geographic grid exceeds lunar bounds')
                    else:
                        notes.append('TIFF CRS/rotation is not a supported Moon 2000 geographic grid; no ground coordinates will be exported.')
                return ImageMetadata(source_format='IMAGE_ONLY', label_location='none', file_name=name,
                    file_size=path.stat().st_size, layout=layout, grid=grid,
                    geometry_status='map_projected' if grid else 'unknown', warnings=notes,
                    original_coordinate_convention=grid.model_dump(include={'frame', 'latitude_type', 'longitude_direction', 'pixel_convention', 'longitude_domain'}) if grid else {}).finish()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in {'JPEG', 'PNG', 'WEBP', 'BMP'}:
                    raise ValueError('Supported containers: JPEG, PNG, TIFF, WebP, BMP')
                width, height = image.size
                container_format = image.format
                if width * height > max_pixels or max(width, height) >= 32767:
                    raise ValueError('Compressed image exceeds decoder budget; locally pre-tile first')
                if image.mode not in {'L', 'RGB', 'RGBA', 'I;16', 'I;16B', 'I;16L'}:
                    raise ValueError('Unsupported image color/sample mode; supply grayscale/RGB or a validated raw label')
                dtype = '>u2' if image.mode == 'I;16B' else '<u2' if image.mode.startswith('I;16') else '|u1'
                bands = len(image.getbands())
                image.verify()
        return ImageMetadata(source_format='IMAGE_ONLY', label_location='none', file_name=name,
            file_size=path.stat().st_size, layout=RasterLayout(width=width, height=height, bands=bands,
                dtype=dtype, sample_type=dtype, decoder='container', container_format=container_format)).finish()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('No usable label: raw IMG requires validated attached PDS3 or associated PDS4 XML; image-only requires a self-describing format') from exc


def inspect_image(path: Path, name: str, mode='auto', xml: bytes | None = None,
                  *, max_pixels=16_777_216, verify_checksum=False, digest=None, provenance: bytes | None = None):
    safe_name(name)
    if mode not in {'auto', 'pds3', 'pds4', 'none'}:
        raise ValueError('Unknown per-image metadata mode')
    if provenance is not None:
        if verify_checksum:
            raise ValueError('Original-file checksum verification requires the original image; a local tile manifest can verify only the uploaded tile SHA-256')
        if xml is not None:
            raise ValueError('Use the local provenance manifest containing the original label, or a direct XML sidecar, not both')
        container = container_metadata(path, name, max_pixels)
        result = apply_manifest(provenance, container, digest or sha256_file(path), mode, DEFAULT_TILE_PIXELS)
        result.selected_mode = mode
        return result
    data, size = header(path), path.stat().st_size
    attached, attached_error = None, None
    if detect_pds3(data):
        try:
            attached = parse_pds3(data, name, size)
        except ValueError as exc:
            attached_error = str(exc)
    container = None
    if not detect_pds3(data):
        try:
            container = container_metadata(path, name, max_pixels)
        except ValueError:
            if xml is None and mode != 'pds3':
                raise
    sidecar, xml_error = None, None
    if xml is not None:
        try:
            sidecar = parse_pds4_raster(xml, name, size, container.layout if container else None)
            if attached_error:
                raise ValueError('PDS4 conflicts with invalid attached PDS3: ' + attached_error)
            if attached:
                a, b = attached.layout, sidecar.layout
                if (a.width, a.height, a.bands, a.dtype, a.offset, a.row_bytes) != (b.width, b.height, b.bands, b.dtype, b.offset, b.row_bytes):
                    raise ValueError('PDS4/PDS3 dimensions, type, offset or stride conflict')
                if attached.product_id and (not sidecar.product_id or sidecar.product_id.split(':')[-1].casefold() != attached.product_id.casefold()):
                    raise ValueError('PDS4/PDS3 product identity conflict')
        except ValueError as exc:
            xml_error = str(exc)
        except Exception as exc:
            # XML ParseError has no guarantee of safe framework response formatting.
            from xml.etree.ElementTree import ParseError
            if not isinstance(exc, ParseError):
                raise
            xml_error = 'Malformed PDS4 XML; supply a valid label for this image'
        if xml_error and mode in {'auto', 'pds4', 'none'}:
            raise ValueError('Supplied XML is invalid: ' + xml_error + '. Correct the XML, or explicitly select Embedded PDS3 after validating its label.')
    if mode == 'pds4' and xml is None:
        raise ValueError('PDS4 XML mode requires a label for this image')
    if mode == 'pds3' or (mode == 'auto' and xml is None and detect_pds3(data)):
        if attached is None:
            raise ValueError('PDS3 attached label validation failed: ' + (attached_error or 'No attached PDS3 label detected'))
        result = attached
        if xml is not None:
            result.warnings.append('Explicit Embedded PDS3 selection overrides supplied XML' + (': ' + xml_error if xml_error else '.'))
    elif mode in {'auto', 'pds4'} and sidecar is not None:
        result = sidecar
    else:
        result = container or container_metadata(path, name, max_pixels)
        if xml is not None:
            result.warnings.append('None / image only explicitly ignores supplied valid XML telemetry.')
    if container and container.grid:
        result.grid, result.geometry_status = container.grid, 'map_projected'
        result.original_coordinate_convention = container.original_coordinate_convention
        result.warnings = [w for w in result.warnings if not w.startswith('Overlap unknown:')]
    if verify_checksum and result.checksum:
        if result.checksum.get('scope') not in {'file', 'image_object'}:
            raise ValueError('Mission checksum verification unavailable: documented checksum scope is not configured')
        md5 = hashlib.md5(usedforsecurity=False)
        with path.open('rb') as stream:
            remaining = result.file_size
            if result.checksum['scope'] == 'image_object':
                stream.seek(result.layout.offset)
                remaining = result.layout.end_byte - result.layout.offset
            while remaining:
                block = stream.read(min(1024 * 1024, remaining))
                if not block:
                    raise ValueError('Truncated checksum data stream')
                md5.update(block)
                remaining -= len(block)
        if md5.hexdigest().lower() != result.checksum['expected'].lower():
            raise ValueError('Label checksum mismatch for declared data scope')
        result.checksum['status'] = 'verified_transport_only'
    result.selected_mode, result.sha256 = mode, digest or sha256_file(path)
    return result.finish()


@dataclass
class RasterTile:
    raw: np.ndarray
    calibrated: np.ndarray
    valid: np.ndarray
    display: np.ndarray
    bytes_read: int


def stretch(values, valid):
    gray = values.astype(np.float64)
    if gray.ndim == 3:
        gray = gray[..., :3].mean(axis=2)
    good = valid & np.isfinite(gray)
    result = np.zeros(gray.shape, dtype=np.uint8)
    if good.any():
        lo, hi = np.percentile(gray[good], [1, 99])
        if hi > lo:
            result[good] = np.rint(np.clip((gray[good] - lo) * (255 / (hi - lo)), 0, 255)).astype(np.uint8)
    return result


def read_tile(path: Path, metadata: ImageMetadata, tile: Tile, *, max_pixels=DEFAULT_TILE_PIXELS):
    layout = metadata.layout
    tile.validate_for(layout, max_pixels)
    layout.validate_size(path.stat().st_size)
    bytes_read = 0
    valid = np.ones((tile.height, tile.width), dtype=bool)
    if layout.decoder == 'raw':
        dtype = np.dtype(layout.dtype)
        raw = np.empty((tile.height, tile.width), dtype=dtype)
        with path.open('rb') as stream:
            for row in range(tile.height):
                position = (layout.offset + (tile.band * layout.height + tile.y + row) * layout.row_bytes
                            + layout.prefix_bytes + tile.x * dtype.itemsize)
                stream.seek(position)
                block = stream.read(tile.width * dtype.itemsize)
                bytes_read += len(block)
                if len(block) != tile.width * dtype.itemsize:
                    raise ValueError('Truncated raster window')
                raw[row] = np.frombuffer(block, dtype=dtype)
        raw = raw.astype(dtype.newbyteorder('='), copy=False)
    else:
        with path.open('rb') as stream:
            magic = stream.read(4)
        if magic in (b'II*\x00', b'MM\x00*', b'II+\x00', b'MM\x00+'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
                with rasterio.open(path, driver='GTiff') as ds:
                    window = Window(tile.x, tile.y, tile.width, tile.height)
                    raw = ds.read(tile.band + 1, window=window)
                    valid &= ds.read_masks(tile.band + 1, window=window) > 0
        else:
            with Image.open(path) as image:
                raw = np.asarray(image.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height)))
                if raw.ndim == 3:
                    if raw.shape[2] == 4:
                        valid &= raw[..., 3] > 0
                    raw = raw[..., :3].mean(axis=2).astype(np.float32)
        bytes_read = raw.nbytes
    valid &= np.isfinite(raw)
    for special in layout.special_constants:
        valid &= raw != special
    if layout.valid_min is not None:
        valid &= raw >= layout.valid_min
    if layout.valid_max is not None:
        valid &= raw <= layout.valid_max
    with np.errstate(over='ignore', invalid='ignore'):
        calibrated = raw.astype(np.float64) * layout.scale + layout.value_offset
    valid &= np.isfinite(calibrated)
    # NaN represents masked physical values, not valid zero. Match stretch is separate.
    calibrated[~valid] = np.nan
    display = np.where(valid, raw, 0).astype(np.uint8) if raw.dtype == np.uint8 and layout.scale == 1 and layout.value_offset == 0 else stretch(calibrated, valid)
    return RasterTile(raw, calibrated, valid, display, bytes_read)


def overlap_for(source: ImageMetadata, reference: ImageMetadata, src_tile: Tile, ref_tile: Tile):
    if source.grid is None or reference.grid is None:
        return 'unknown'
    a, b = source.grid.window(src_tile), reference.grid.window(ref_tile)
    top, bottom = min(a.north_lat, b.north_lat), max(a.north_lat - src_tile.height * a.pixel_size_deg, b.north_lat - ref_tile.height * b.pixel_size_deg)
    longitude = any(max(a.west_lon, b.west_lon + shift) < min(a.west_lon + src_tile.width * a.pixel_size_deg, b.west_lon + shift + ref_tile.width * b.pixel_size_deg) for shift in (-360, 0, 360))
    return 'verified_grid_overlap' if top > bottom and longitude else 'nonoverlap'
