"""Bounded ODL subset; no evaluation, includes, detached paths, or network access.

ODL 2.1 text strings do not permit a quote delimiter inside a quoted string
(PDS3 chapter 12 §12.3.3.1). Backslash/doubled-quote dialects are rejected.
Unknown syntactic constructs fail rather than being interpreted heuristically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from pathlib import Path
import re

from .metadata import HEADER_LIMIT, ImageMetadata, RasterLayout, checked, finite, safe_name


class RadixInt(int):
    """Keep the lexical origin so float bit-pattern specials cannot be guessed."""


@dataclass(frozen=True)
class Quantity:
    value: object
    unit: str


@dataclass(frozen=True)
class Sentinel:
    name: str


@dataclass
class Node:
    kind: str
    name: str
    values: dict[str, list] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)

    def get(self, key, default=None):
        values = self.values.get(key.upper(), [])
        if len(values) > 1:
            raise ValueError(f"Ambiguous repeated {key} in {self.kind}={self.name}")
        return values[0] if values else default


def _tokens(text):
    pos, count = 0, 0
    while pos < len(text):
        if text[pos].isspace() or text[pos] == ';':
            pos += 1
            continue
        if text.startswith('/*', pos):
            end = text.find('*/', pos + 2)
            if end < 0:
                raise ValueError("Unterminated ODL comment / header limit exceeded")
            pos = end + 2
            continue
        if text[pos] == '&':
            match = re.match(r'&[ \t]*\r?\n', text[pos:])
            if not match:
                raise ValueError("Unsupported ODL continuation")
            pos += len(match[0])
            continue
        count += 1
        if count > 65536:
            raise ValueError("ODL token quota exceeded")
        char = text[pos]
        if char in '\"\'':
            end = text.find(char, pos + 1)
            if end < 0:
                raise ValueError("Unterminated ODL string / header limit exceeded")
            value = text[pos + 1:end]
            if char == "'" and ('\n' in value or '\r' in value):
                raise ValueError("ODL symbol strings cannot span lines")
            if '\\' in value or any(ord(c) > 127 for c in value):
                raise ValueError("Unsupported ODL string escape/encoding")
            yield ('STRING', value if char == '"' else value.upper(), end + 1)
            pos = end + 1
        elif char in '=(),<>':
            yield (char, char, pos + 1)
            pos += 1
        else:
            match = re.match(r'[^\s=(),<>;"\'&]+', text[pos:])
            if not match:
                raise ValueError("Unsupported ODL syntax")
            value = match[0]
            yield ('WORD', value, pos + len(value))
            pos += len(value)


def parse_odl(data: bytes, max_header=HEADER_LIMIT) -> tuple[Node, int]:
    if len(data) > max_header:
        data = data[:max_header]
    stream = iter(_tokens(data.decode('latin1')))
    current = next(stream, None)

    def take(kind=None):
        nonlocal current
        token = current
        if token is None or (kind and token[0] != kind):
            raise ValueError("Malformed ODL assignment or header limit exceeded")
        current = next(stream, None)
        return token

    def value(depth=0):
        if depth > 16:
            raise ValueError("ODL sequence nesting limit exceeded")
        if current and current[0] == '(':
            take('(')
            values = []
            if current and current[0] != ')':
                while True:
                    values.append(value(depth + 1))
                    if len(values) > 4096:
                        raise ValueError("ODL sequence quota exceeded")
                    if not current or current[0] != ',':
                        break
                    take(',')
            take(')')
            result = tuple(values)
        else:
            kind, text, _ = take()
            if kind == 'STRING':
                result = text
            elif kind != 'WORD':
                raise ValueError("Unsupported ODL value")
            elif text.upper() in {'NULL', 'UNK', 'N/A'}:
                result = Sentinel(text.upper())
            elif re.fullmatch(r'\d+#[+-]?[0-9A-Fa-f]+#', text):
                radix, digits, _ = text.split('#')
                if not 2 <= int(radix) <= 16:
                    raise ValueError("ODL radix must be 2..16")
                if len(digits) > 24:
                    raise ValueError('ODL radix literal exceeds supported precision')
                result = RadixInt(int(digits, int(radix)))
            elif re.fullmatch(r'[+-]?\d+', text):
                if len(text) > 24:
                    raise ValueError("ODL integer exceeds supported precision")
                result = int(text)
            elif re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?', text):
                result = finite(text, 'ODL number')
            elif re.fullmatch(r'[A-Za-z][A-Za-z0-9_:\-./+]*|\d{4}-[0-9TtZz:.+\-]+', text):
                result = text
            else:
                raise ValueError("Unsupported ODL scalar syntax")
        if current and current[0] == '<':
            take('<')
            units = []
            while current and current[0] != '>':
                units.append(take('WORD')[1])
                if len(units) > 16:
                    raise ValueError("ODL units quota exceeded")
            take('>')
            if not units or isinstance(result, (str, tuple, Sentinel, Quantity)):
                raise ValueError("Invalid ODL units expression")
            result = Quantity(result, ' '.join(units).upper())
        return result

    root = Node('ROOT', 'ROOT')
    stack = [root]
    while current:
        # Do not lex binary data after END. END is valid only at statement scope.
        if current[0] == 'WORD' and current[1].upper() == 'END':
            if len(stack) != 1:
                raise ValueError("Unclosed ODL OBJECT/GROUP")
            if current[2] < len(data) and chr(data[current[2]]) not in ' \t\r\n\f\v;':
                raise ValueError('END must be a standalone token')
            return root, current[2]
        key = take('WORD')[1].upper()
        if not re.fullmatch(r'\^?[A-Z][A-Z0-9_]*(?::[A-Z][A-Z0-9_]*)?', key):
            raise ValueError("Unsupported ODL keyword")
        if key in {'^STRUCTURE', '^INCLUDE', 'INCLUDE'}:
            raise ValueError("ODL external includes are forbidden")
        if key in {'END_OBJECT', 'END_GROUP'} and (current is None or current[0] != '='):
            if len(stack) == 1 or stack[-1].kind != key[4:]:
                raise ValueError("Mismatched ODL END_OBJECT/END_GROUP")
            stack.pop()
            continue
        take('=')
        parsed = value()
        if key in {'OBJECT', 'GROUP'}:
            if not isinstance(parsed, str) or len(stack) >= 32:
                raise ValueError("Invalid or excessive ODL object nesting")
            child = Node(key, parsed.upper())
            stack[-1].children.append(child)
            stack.append(child)
        elif key in {'END_OBJECT', 'END_GROUP'}:
            if len(stack) == 1 or stack[-1].kind != key[4:] or stack[-1].name != str(parsed).upper():
                raise ValueError("Mismatched ODL END_OBJECT/END_GROUP")
            stack.pop()
        else:
            stack[-1].values.setdefault(key, []).append(parsed)
    raise ValueError("Standalone END not found within the configured header limit")


def detect_pds3(data: bytes):
    # Lexical detection ignores comments/strings; full parsing proves root scope.
    try:
        tokens = iter(_tokens(data[:4096].decode('latin1')))
        for kind, text, _ in tokens:
            if kind == 'WORD' and text.upper() == 'PDS_VERSION_ID':
                eq, val = next(tokens, None), next(tokens, None)
                return bool(eq and eq[0] == '=' and val and val[1].upper() == 'PDS3')
    except ValueError:
        pass
    return False


def integer(value, name, minimum=1):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    checked(int(value))
    return int(value)


def image_offset(pointer, record_bytes):
    if isinstance(pointer, (str, tuple)):
        raise ValueError("Detached ^IMAGE pointers are unsupported; use an attached label or an explicitly associated PDS4 XML")
    if isinstance(pointer, Quantity):
        if pointer.unit not in {'BYTE', 'BYTES'}:
            raise ValueError("^IMAGE pointer units must be BYTES")
        return integer(pointer.value, '^IMAGE') - 1
    return checked(integer(pointer, '^IMAGE') - 1, integer(record_bytes, 'RECORD_BYTES'))


def physical_type(sample_type, bits):
    integers = {'LSB_INTEGER': '<i', 'MSB_INTEGER': '>i', 'INTEGER': '>i',
                'LSB_UNSIGNED_INTEGER': '<u', 'MSB_UNSIGNED_INTEGER': '>u', 'UNSIGNED_INTEGER': '>u'}
    floats = {'IEEE_REAL': '>f', 'PC_REAL': '<f'}
    sample_type = str(sample_type).upper()
    if sample_type in integers and bits in (8, 16, 32):
        return ('|' + integers[sample_type][-1] if bits == 8 else integers[sample_type]) + str(bits // 8)
    if sample_type in floats and bits in (32, 64):
        return floats[sample_type] + str(bits // 8)
    raise ValueError(f"Unsupported PDS3 SAMPLE_TYPE/SAMPLE_BITS: {sample_type}/{bits}")


def parse_pds3(data: bytes, file_name: str, file_size: int) -> ImageMetadata:
    safe_name(file_name)
    root, end = parse_odl(data)
    if str(root.get('PDS_VERSION_ID')).upper() != 'PDS3':
        raise ValueError("Attached label must declare PDS_VERSION_ID=PDS3")
    record_type = str(root.get('RECORD_TYPE')).upper()
    if record_type not in {'FIXED_LENGTH', 'UNDEFINED'}:
        raise ValueError("Only FIXED_LENGTH or UNDEFINED binary records are supported")
    record_bytes = root.get('RECORD_BYTES')
    if record_bytes is not None:
        integer(record_bytes, 'RECORD_BYTES')
    if record_type == 'FIXED_LENGTH' and record_bytes is None:
        raise ValueError("FIXED_LENGTH requires RECORD_BYTES")
    label_records, file_records = root.get('LABEL_RECORDS'), root.get('FILE_RECORDS')
    label_end = end
    if label_records is not None:
        label_end = checked(integer(label_records, 'LABEL_RECORDS'), integer(record_bytes, 'RECORD_BYTES'))
        if label_end > HEADER_LIMIT or end > label_end or label_end > file_size:
            raise ValueError("Header does not fit LABEL_RECORDS * RECORD_BYTES / header limit")
    if file_records is not None:
        declared = checked(integer(file_records, 'FILE_RECORDS'), integer(record_bytes, 'RECORD_BYTES'))
        if declared != file_size:
            raise ValueError("FILE_RECORDS * RECORD_BYTES does not match actual file size")
    offset = image_offset(root.get('^IMAGE'), record_bytes)
    if offset < label_end:
        raise ValueError("^IMAGE overlaps attached label records")
    images = [n for n in root.children if n.kind == 'OBJECT' and n.name == 'IMAGE']
    if len(images) != 1:
        raise ValueError("Exactly one top-level OBJECT=IMAGE is required")
    image = images[0]
    if str(image.get('ENCODING_TYPE', 'N/A')).upper() not in {'N/A', 'NONE', 'UNCOMPRESSED'}:
        raise ValueError("Compressed PDS3 images require a separately validated mission decoder")
    bands = integer(image.get('BANDS', 1), 'BANDS')
    if str(image.get('BAND_STORAGE_TYPE', 'BAND_SEQUENTIAL' if bands == 1 else None)).upper() != 'BAND_SEQUENTIAL':
        raise ValueError("Only BAND_SEQUENTIAL band layout is supported")
    sample_type = str(image.get('SAMPLE_TYPE')).upper()
    specials = []
    for key in ('NULL', 'MISSING_CONSTANT', 'INVALID_CONSTANT', 'CORE_NULL', 'LOW_REPR_SATURATION',
                'LOW_INSTR_SATURATION', 'HIGH_REPR_SATURATION', 'HIGH_INSTR_SATURATION'):
        raw = image.get(key)
        if raw is not None and not isinstance(raw, Sentinel):
            if sample_type in {'IEEE_REAL', 'PC_REAL'} and isinstance(raw, RadixInt):
                raise ValueError('Radix-encoded float special bit patterns require a mission decoder')
            specials.append(finite(raw, key))
    def number(key, default):
        raw = image.get(key, default)
        return None if raw is None or isinstance(raw, Sentinel) else finite(raw, key)
    layout = RasterLayout(width=integer(image.get('LINE_SAMPLES'), 'LINE_SAMPLES'),
        height=integer(image.get('LINES'), 'LINES'), bands=bands,
        dtype=physical_type(sample_type, integer(image.get('SAMPLE_BITS'), 'SAMPLE_BITS')),
        sample_type=sample_type, offset=offset,
        prefix_bytes=integer(image.get('LINE_PREFIX_BYTES', 0), 'LINE_PREFIX_BYTES', 0),
        suffix_bytes=integer(image.get('LINE_SUFFIX_BYTES', 0), 'LINE_SUFFIX_BYTES', 0),
        scale=number('SCALING_FACTOR', 1.0), value_offset=number('OFFSET', 0.0),
        special_constants=specials, valid_min=number('VALID_MINIMUM', None), valid_max=number('VALID_MAXIMUM', None))
    layout.validate_size(file_size)
    def text(key):
        val = root.get(key)
        return val if isinstance(val, str) else None
    product, dataset = text('PRODUCT_ID'), text('DATA_SET_ID')
    target, frame, level = text('TARGET_NAME'), text('FRAME_ID'), text('PRODUCT_TYPE')
    acquisition = {k: text(k) for k in ('START_TIME', 'STOP_TIME', 'SPACECRAFT_CLOCK_START_COUNT', 'SPACECRAFT_CLOCK_STOP_COUNT') if text(k)}
    warnings = []
    if target and target.upper() != 'MOON':
        raise ValueError("Lunar registration requires TARGET_NAME=MOON")
    if dataset and 'LROC' in dataset.upper():
        if target != 'MOON' or not product or not level or not frame or not acquisition.get('START_TIME') or not acquisition.get('STOP_TIME'):
            raise ValueError("LROC profile requires Moon, DATA_SET_ID, PRODUCT_ID, PRODUCT_TYPE, FRAME_ID and START/STOP_TIME")
        if not re.fullmatch(r'LRO-L-LROC-[23]-(EDR|CDR)-V\d+(?:\.\d+)*', dataset, re.I):
            raise ValueError("Unsupported LROC dataset profile (EDR/CDR required)")
        expected = 'CDR' if '-CDR-' in dataset.upper() else 'EDR'
        if level.upper() != expected or not re.fullmatch(r'M\d+[LR][EC]', product, re.I):
            raise ValueError("Inconsistent LROC product identity or processing level")
        if Path(file_name).stem.casefold() != product.casefold():
            raise ValueError("LROC PRODUCT_ID must match the uploaded image filename")
        if frame.upper() not in {'LEFT', 'RIGHT'} or product[-2].upper() != frame[0].upper() or product[-1].upper() != expected[0]:
            raise ValueError("LROC FRAME_ID/product identity conflict")
        try:
            times = [datetime.fromisoformat(acquisition[k].replace('Z', '+00:00')) for k in ('START_TIME', 'STOP_TIME')]
            if times[1] < times[0]:
                raise ValueError('Stop precedes start')
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid or unsupported LROC acquisition time interval') from exc
        if image.get('PRODUCT_ID') is not None and image.get('PRODUCT_ID') != product:
            raise ValueError('LROC IMAGE product identity conflicts with root')
        if layout.width > 5064 or bands != 1:
            raise ValueError("Invalid LROC NAC dimensions/bands")
        warnings.append("LROC EDR/CDR decoding provides no line-to-Moon camera transform; ISIS/SPICE is external and optional.")
    checksum = {}
    md5 = image.get('MD5_CHECKSUM')
    if md5 is not None and dataset and 'LROC' in dataset.upper():
        if not isinstance(md5, str) or not re.fullmatch(r'[a-fA-F0-9]{32}', md5):
            raise ValueError('Invalid LROC image data-stream MD5 checksum')
        checksum = {'algorithm':'MD5', 'expected':md5, 'scope':'image_object', 'status':'not_verified',
                    'specification':'LROC EDR/CDR SIS v1.18 section 3.3: IMAGE data stream'}
        warnings.append('LROC IMAGE data-stream MD5 is not verified unless requested; MD5 is not an authenticity guarantee.')
    elif root.get('MD5_CHECKSUM') is not None or md5 is not None:
        checksum = {'algorithm': 'MD5', 'status': 'not_verified', 'reason': 'Mission checksum scope not configured; not a security guarantee'}
        warnings.append('Label MD5 not verified: documented mission scope is not configured.')
    return ImageMetadata(source_format='PDS3', label_location='attached', file_name=file_name, file_size=file_size,
        product_id=product, data_set_id=dataset, instrument=text('INSTRUMENT_ID') or text('INSTRUMENT_NAME'),
        processing_level=level, target=target, frame_id=frame, acquisition=acquisition, layout=layout,
        label_sha256=hashlib.sha256(data[:end]).hexdigest(), checksum=checksum, warnings=warnings,
        field_provenance={k: 'attached PDS3 root' for k in ('product_id', 'data_set_id', 'instrument', 'processing_level', 'target', 'frame_id')},
        units={'pixels': 'raw DN', 'scale': 'label SCALING_FACTOR', 'offset': 'label OFFSET'}).finish()
