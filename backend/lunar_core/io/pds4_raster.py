"""Safe PDS4 file/array association and raw physical layout validation."""
from __future__ import annotations

import hashlib
import re

from .metadata import ImageMetadata, RasterLayout, finite, safe_name
from .pds4_parser import _get_xml_root, parse_metadata

NS = '{http://pds.nasa.gov/pds4/pds/v1}'
TYPES = {'UnsignedByte': '|u1', 'SignedByte': '|i1',
         **{f'{sign}{endian}{size}': prefix + kind + str(size)
            for sign, kind in [('Unsigned', 'u'), ('Signed', 'i')]
            for endian, prefix in [('LSB', '<'), ('MSB', '>')] for size in (2, 4)},
         **{f'IEEE754{endian}{name}': prefix + 'f' + str(size)
            for endian, prefix in [('LSB', '<'), ('MSB', '>')]
            for name, size in [('Single', 4), ('Double', 8)]}}


def one(parent, tag, required=True):
    matches = parent.findall(NS + tag)
    if len(matches) != 1:
        if not matches and not required:
            return None
        raise ValueError(f'PDS4 requires exactly one {tag}')
    return matches[0]


def content(parent, tag, default=None):
    el = one(parent, tag, default is None)
    return (el.text or '').strip() if el is not None else default


def uint(text, name):
    if not re.fullmatch(r'\d{1,19}', text):
        raise ValueError(f'Invalid PDS4 integer: {name}')
    return int(text)


def parse_pds4_raster(data: bytes, file_name: str, file_size: int,
                     container_layout: RasterLayout | None = None) -> ImageMetadata:
    root = _get_xml_root(data)
    safe_name(file_name)
    areas = root.findall(NS + 'File_Area_Observational') + root.findall(NS + 'File_Area_Observational_Supplemental') + root.findall(NS + 'File_Area_Encoded_Image')
    if len(areas) != 1:
        raise ValueError('PDS4 must have one File_Area_Observational identifying this upload')
    area = areas[0]
    file = one(area, 'File')
    name = safe_name(content(file, 'file_name'))
    if name.casefold() != file_name.casefold():
        raise ValueError('PDS4 file_name does not match selected image; associate the correct XML')
    size = one(file, 'file_size', False)
    if size is not None:
        if size.get('unit', 'byte').lower() not in {'byte', 'bytes'} or uint(size.text or '', 'file_size') != file_size:
            raise ValueError('PDS4 file_size does not match uploaded bytes')
    encoded = one(area, 'Encoded_Image', False)
    if encoded is not None:
        if container_layout is None:
            raise ValueError('PDS4 Encoded_Image requires a validated self-describing image')
        if content(encoded, 'encoding_standard_id') != container_layout.container_format:
            raise ValueError('PDS4 encoding_standard_id conflicts with actual file format')
        offset = one(encoded, 'offset')
        if offset.get('unit', 'byte').lower() not in {'byte', 'bytes'} or uint(offset.text or '', 'offset') != 0:
            raise ValueError('Encoded image must start at byte zero')
        length = one(encoded, 'object_length', False)
        if length is not None and (length.get('unit','byte').lower() not in {'byte','bytes'} or uint(length.text or '', 'object_length') != file_size):
            raise ValueError('Encoded image object_length differs from uploaded file size')
        if any(el.tag in {NS+'Array_2D_Image', NS+'Array_3D_Image'} for el in area):
            raise ValueError('Ambiguous PDS4 array/encoded descriptors')
        layout = container_layout.model_copy(deep=True)
    else:
        if container_layout is not None:
            raise ValueError('Raw PDS4 array descriptor cannot describe an encoded image; use its correct Encoded_Image label')
        arrays = [el for el in area if el.tag in {NS + 'Array_2D_Image', NS + 'Array_3D_Image'}]
        if len(arrays) != 1:
            raise ValueError('Exactly one PDS4 Array_2D_Image/Array_3D_Image is supported')
        array = arrays[0]
        offset = one(array, 'offset')
        if offset.get('unit', 'byte').lower() not in {'byte', 'bytes'}:
            raise ValueError('PDS4 offset requires byte units')
        offset_value = uint(offset.text or '', 'offset')
        if content(array, 'axis_index_order') != 'Last Index Fastest':
            raise ValueError('Only PDS4 Last Index Fastest is supported')
        axes = []
        for axis in array.findall(NS + 'Axis_Array'):
            axes.append((uint(content(axis, 'sequence_number'), 'sequence_number'), content(axis, 'axis_name'), uint(content(axis, 'elements'), 'elements')))
        axes.sort()
        if (len(axes) != uint(content(array, 'axes'), 'axes')
                or [a[0] for a in axes] != list(range(1, len(axes) + 1))
                or [a[1] for a in axes] not in [['Line', 'Sample'], ['Band', 'Line', 'Sample']]):
            raise ValueError('Unsupported/ambiguous PDS4 axes; expected [Band,] Line, Sample in storage order')
        element = one(array, 'Element_Array')
        sample_type = content(element, 'data_type')
        if sample_type not in TYPES:
            raise ValueError(f'Unsupported PDS4 data_type: {sample_type}')
        special = one(array, 'Special_Constants', False)
        constants = []
        limits = {}
        if special is not None:
            for child in special:
                key = child.tag.removeprefix(NS)
                if key in {'valid_minimum', 'valid_maximum'}:
                    limits['valid_min' if key == 'valid_minimum' else 'valid_max'] = finite(child.text, key)
                elif key.endswith('_constant') or 'saturation' in key:
                    constants.append(finite(child.text, key))
                else:
                    raise ValueError(f'Unsupported PDS4 Special_Constants field: {key}')
        layout = RasterLayout(width=axes[-1][2], height=axes[-2][2], bands=axes[0][2] if len(axes) == 3 else 1,
            dtype=TYPES[sample_type], sample_type=sample_type, offset=offset_value,
            scale=finite(content(element, 'scaling_factor', '1'), 'scaling_factor'),
            value_offset=finite(content(element, 'value_offset', '0'), 'value_offset'), special_constants=constants, **limits)
        layout.validate_size(file_size)
    # Optional acquisition values remain optional. Four corners never become a grid.
    telemetry = parse_metadata(data)
    fields = {k: getattr(telemetry, k) for k in ('product_id', 'instrument', 'gsd_meters', 'incidence_angle_deg', 'emission_angle_deg', 'solar_azimuth_deg')}
    warnings = ['Acquisition coordinates/corners, if supplied, are not a validated pixel-to-ground transform.']
    def local_text(name):
        values = [(e.text or '').strip() for e in root.iter() if e.tag.rsplit('}', 1)[-1] == name]
        if len(set(values)) > 1:
            raise ValueError(f'Ambiguous PDS4 {name}')
        return values[0] if values else None
    target = local_text('target_name')
    target_nodes = root.findall('.//' + NS + 'Target_Identification/' + NS + 'name')
    targets = {(e.text or '').strip().casefold() for e in target_nodes}
    if target:
        targets.add(target.casefold())
    if len(targets) > 1:
        raise ValueError('Ambiguous PDS4 target identification')
    target = target or (next(iter(targets)) if targets else None)
    if target and target.casefold() != 'moon':
        raise ValueError('Lunar registration requires Moon target')
    # Prefer a raw product identifier over a logical identifier when both are present.
    fields['product_id'] = local_text('product_id') or telemetry.product_id
    checksum = {}
    md5 = one(file, 'md5_checksum', False)
    if md5 is not None:
        checksum = {'algorithm': 'MD5', 'expected': (md5.text or '').strip(), 'scope': 'file', 'status': 'not_verified'}
        if not re.fullmatch(r'[a-fA-F0-9]{32}', checksum['expected']):
            raise ValueError('Malformed PDS4 file MD5 checksum')
        warnings.append('MD5 is a transport checksum, not an authenticity/security guarantee.')
    return ImageMetadata(source_format='PDS4', label_location='xml_sidecar', file_name=file_name, file_size=file_size,
        layout=layout, label_sha256=hashlib.sha256(data).hexdigest(), target=target,
        processing_level=local_text('processing_level'), checksum=checksum, warnings=warnings,
        field_provenance={k: 'PDS4 XML explicitly supplied field' for k, v in fields.items() if v is not None},
        units={'gsd_meters': 'm/pixel', 'angles': 'degrees'}, **fields).finish()
