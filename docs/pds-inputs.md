# Mixed PDS3/PDS4 inputs and bounded registration

This implementation registers selected image tiles. It does not establish that arbitrary LROC and Chandrayaan observations overlap, and feature residuals do not measure independent lunar ground accuracy.

## Using the workspace

1. Start the interface and processing service, then enter the service URL in **Settings**. The initial workspace is empty. **Load demo data** is the only way to load simulated metadata/results.
2. Load the source and reference independently. Each card has **Auto**, **PDS4 XML**, **Embedded PDS3**, and **None / image only**.
3. In Auto, an explicitly supplied XML wins only after validation. Without XML, a valid attached PDS3 label is used; otherwise a self-describing image can use image-only mode. TIFF, PNG, JPEG, WebP and BMP are supported within decoder limits. A raw IMG cannot use image-only mode.
4. Use **Inspect image and label** on each card. This sends that file to the server. The browser does not claim authoritative metadata. Inspect the label location, product, processing level, full dimensions, type, available geometry, missing fields and warnings.
5. Select zero-based `x`, `y`, `width`, `height`, and `band`. The default registration budget is 1,048,576 pixels per tile, with at least 32 pixels on each axis. Use **Validate selected tile** to get a server-decoded preview. Coordinates displayed under the card include a local extraction's original full-scene origin when present.
6. When overlap is unknown, choose corresponding tiles using your observation context, a trusted external map or camera workflow. Confirm **I selected corresponding tiles** before registration. Do not choose the two example products merely because they are lunar images.
7. Run registration, inspect correspondences and quality, then export. Unknown geometry always keeps the result `review_required`, including when the numerical feature gates pass. A rejected pair still supplies a dossier; it cannot supply a falsely successful warped raster.

A supplied invalid/mismatched XML causes an error in Auto/PDS4 mode. Correct it, or deliberately select **Embedded PDS3**. That override validates the attached label independently and retains the XML conflict warning in the report. The other card's metadata choice is unaffected.

Changing a file, XML, manifest, mode, tile or service invalidates its inspection and all previous registration artifacts. Changing only the tile retains validated header metadata but requires tile revalidation. Clearing/loading demo data cancels active work and resets stale results. Parsing errors use live status/error announcements; controls use native keyboard-accessible inputs.

## Large scenes: supported local extraction

The browser cap stays **512 MiB per file**. The LROC layout in the prompt occupies 528,929,736 bytes (approximately 504.43 MiB), including its label. It fits that file cap, but not a full-scene decode or registration. The TMC-2 layout occupies **1,184,864,000 bytes** and exceeds the browser cap. Use local extraction for TMC-2 and preferably for either large scene. Inspection, tile preview and submission otherwise retransmit the bounded original file; slow connections can hit the 120-second upload deadline.

From the repository root, after installing `backend/requirements.txt`:

```powershell
# Choose actual corresponding windows; these origins are only command examples.
python -m backend.tile_cli M174353756RC.IMG --x 1200 --y 18000 --width 512 --height 512 --output work/lroc-tile.tif
python -m backend.tile_cli observation.img --xml observation.xml --x 900 --y 72000 --width 512 --height 512 --output work/tmc-tile.tif
```

For each image card, load the generated TIFF and then its matching `.provenance.json` under **Large scene / local tile provenance**. Do not attach the original raw-file XML directly to the TIFF: that XML describes another byte layout. The manifest carries the original label and its standard, original product identity, full dimensions, original SHA-256, selected band and origin. The server validates the tile hash, contained label, bounds and numeric type.

The original full-file hash and extraction correspondence remain **local-tool claims** because the server has not received the original scene. The dossier explicitly distinguishes these from the server-verified tile bytes. This is provenance, not cryptographic authentication. Manifest data cannot provide a geotransform. Only an independently supplied and validated geographic lunar TIFF grid can establish that transform.

The extractor seeks selected row segments only. It also computes SHA-256 in a sequential 1 MiB streaming pass over the original file; that pass reads the whole file without decoding it or keeping it in RAM. Existing outputs are refused. No label-named paths or URLs are opened.

## Supported layout semantics

PDS3 headers are bounded to 1 MiB, detected from the first 4096 bytes, and parsed without evaluation. `END` is recognized at statement scope outside comments/quotes. Record pointers are one-based; byte pointers are one-based. `LABEL_RECORDS * RECORD_BYTES` must contain the header, and the declared image must fit the actual file. `FILE_RECORDS`, when declared, must match the actual byte count. A physical record is **not** an image row.

For the public layout subset in `backend/tests/fixtures/public_lroc_layout.odl`, the first pixel is at byte **5064**; a line has **5064 int16 samples**, hence **10128 bytes**. Its physical row spans two records. The fixture omits non-layout mission fields and contains no image data.

The ODL subset supports case-insensitive and namespace keywords, whitespace/CRLF/LF, block comments, multiline text, `&` line continuations, scalar integers/finite reals, radix integers, units, ordered sequences, scoped repeated keys, and matching nested OBJECT/GROUP delimiters. Relevant duplicate fields are rejected as ambiguous. Literal `NULL`/`UNK`/`N/A` are distinct from numeric pixel special values. ODL 2.1 does not permit embedded quote delimiters in text strings; backslash/doubled-quote dialects are explicitly rejected.

Allowlisted PDS3 physical types: signed/unsigned 8/16/32-bit little/big-endian integers, `INTEGER`/`UNSIGNED_INTEGER` big-endian aliases, and 32/64-bit `IEEE_REAL` (big-endian) / `PC_REAL` (little-endian). Single-band and explicitly band-sequential multi-band arrays are supported, with a selected zero-based band. Prefix/suffix bytes are included in row offsets. Packed, compressed, complex, VAX/unknown float encodings, float special constants encoded as radix bit patterns, unknown interleave, detached `.LBL`, and external includes fail explicitly.

The LROC NAC EDR/CDR profile checks dataset, product identity, filename association, processing level, left/right frame, Moon target, supported time interval, image identity and dimension consistency. Generic PDS3 labels do not inherit LROC naming rules. Missing geometric fields remain unknown. If checksum verification is requested, LROC IMAGE MD5 covers the image data stream at the declared object offset/length (SIS 1.18 §3.3); PDS4 File MD5 covers the file. Other unspecified mission checksum scopes are refused. MD5 is only a transport check. Server-computed SHA-256 records byte provenance.

PDS4 accepts one associated raw `Array_2D_Image` or band-sequential `Array_3D_Image`, ordered `[Band,] Line, Sample` with `Last Index Fastest`, explicit byte offset and allowlisted element type. `UnsignedLSB2` is uint16, not int16. A correctly associated `Encoded_Image` can describe a supported container with offset zero and matching encoding. Raw array descriptors cannot be substituted for encoded-image labels. File names are compared as safe plain names and never used as storage paths. File sizes and optional checksum syntax are validated. XML is limited to 1 MiB, 20,000 elements and depth 48; DTDs/entities/non-UTF8 encodings are refused. Schema references and processing-instruction URLs are never fetched. Mission telemetry aliases are limited; an unrecognized geometry field remains absent rather than estimated.

Masks are evaluated on raw samples before scaling/offset, including null/saturation/range checks and non-finite floats. Valid zero remains valid. Decoded DN, float64 calibration and 8-bit display/matching stretch are separate. Entropy/Wallis statistics use explicit masks; matcher candidates/descriptors exclude invalid regions. The registered TIFF stores float64 interpolated raw DN with scale/offset tags and an internal validity mask. Bilinear output support must be fully valid and intersect valid reference pixels.

## Geometry and scientific limits

An attached NAC CDR label decodes samples; it does not locate every line on the Moon. Acquisition angles, GSD, bounding summaries, camera models and map grids are distinct. Four corners never become a north-up affine transform.

The supported map path is a TIFF whose CRS is semantically equivalent to **Moon 2000 / IAU2000:30100**, spherical radius 1,737,400 m, longitude/latitude degrees, east-positive longitude, and a north-up square-pixel affine. The record declares planetocentric latitude (equivalent to planetographic on this sphere), continuous longitude and upper-left pixel corners. Tie points use zero-based pixel centers. Reference tile offsets update the grid. Geographic rectangle overlap handles 0/360 and antimeridian crossing; disjoint selected grids are rejected. Bounding summaries are not used to establish overlap. Rotated/projected TIFF grids and other lunar frames are currently not reused for ground export.

If both GSD and incidence fields are present in associated labels, the existing adaptive matcher route is retained. Otherwise the documented fallback is SIFT/FLANN with rationale **insufficient telemetry**; the API can explicitly request LightGlue. No numeric telemetry is invented. Legacy JSON telemetry is recorded as unverified and does not drive geometry routing. An absent label on one side does not erase the other's record.

Only a validated reference grid enables a georeferenced TIFF and CSV latitude/longitude. Otherwise the TIFF has no CRS and those CSV fields are empty. The CSV has tile-local `src_x/src_y/ref_x/ref_y` and `src_full_x/src_full_y/ref_full_x/ref_full_y`; full coordinates include both extraction and selection offsets. The dossier records the translation matrices, matching mode, label standards, product IDs, layout, masks, warnings, geometry source, quality basis and origins. Even a valid map grid is **not independent ground validation**; that field remains false.

ISIS/SPICE integration is not implemented or bundled. For an optional external NAC CDR workflow, USGS directs import through `pds2isis`, then mission-appropriate `spiceinit` and `cam2map`. Treat **ISIS 7.1.0** as the documented reference version, not an installed dependency. The upstream [USGS software inventory](https://github.com/DOI-USGS/ISIS3/blob/dev/code.json) identifies ISIS as Public Domain / CC0-1.0; check the license and third-party notices shipped with the exact external installation. No kernels/body-shape data are distributed, downloaded or implicitly selected; record their exact versions/checksums, frame, shape and license/access terms in any independently processed product. Missing camera/kernel support stays explicit and cannot be replaced by a bounding box.

## Reproducible checks

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:TORCH_HOME=Join-Path (Get-Location) 'work/torch'
python -m pytest backend/tests -q
python -m backend.benchmark_tiles --output work/pds-tile-benchmark.json
node --test backend/tests/registration-client.test.mjs
pnpm exec tsc --noEmit
pnpm build
```

The neural tests require the separately pinned LightGlue/PyTorch requirements and cached weights described in the repository README. Core new PDS/SIFT tests do not download weights. For browser checks, install Playwright **1.62.1** as a development tool and its Chromium binary, start `pnpm dev`, then run `node --test backend/tests/workspace-ui.test.mjs`. An existing Edge installation can be selected with `LUNARGRID_BROWSER_CHANNEL=msedge`. `LUNARGRID_PLAYWRIGHT_MODULE` accepts the module URL of an already installed Playwright package; `LUNARGRID_UI_URL` changes the test server URL.

The browser test exercises real React controls/state with mocked service responses. Backend tests separately exercise real multipart ingestion, matching, raster decoding and artifact export. The new known-overlap mixed-standard test uses synthetic distributed texture with a known (-6,-4) translation. It makes no claim about overlap or accuracy of the two mission examples.

The bounded-read benchmark simulates seekable files with the exact full LROC and TMC-2 layouts, including the last 512×512 window. Each reads **524,288 pixel bytes in 512 row seeks**, with approximately **9.3 MiB traced allocation** on this workstation. It asserts read bounds and a 64 MiB allocation ceiling. This measures layout/read volume and Python/NumPy allocations, not real-disk throughput, whole-process peak RSS, network capacity or deployment performance. Use the local extractor for real-product timing and validate service/container limits before public deployment.

## Authoritative references

- [NASA ODL chapter 12](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter12.pdf), [pointers chapter 14](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter14.pdf), [physical records chapter 15](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter15.pdf), [physical sample types chapter 3](https://pds.nasa.gov/datastandards/pds3/standards/sr/Chapter03.pdf).
- [LROC EDR/CDR SIS 1.18 archive mirror](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DOCUMENT/LROCSIS.PDF), consulted for NAC frame/product meanings, special pixels and image-stream checksums. Its historical examples do not override the uploaded label's dimensions/scaling or actual byte count.
- [PDS4 array storage order](https://pds.nasa.gov/datastandards/documents/dd/v1/PDS4_PDS_DD_1O00/webhelp/all/ch05s29.html), [encoded images](https://pds.nasa.gov/datastandards/documents/dd/common/current/ch03s39.html).
- [ISIS NAC import guidance](https://isis.astrogeology.usgs.gov/dev/Application/presentation/Tabbed/lronac2isis/lronac2isis.html), [SPICE initialization](https://isis.astrogeology.usgs.gov/dev/Application/presentation/Tabbed/spiceinit/spiceinit.html), [ISIS 7.1.0 camera-to-map](https://isis.astrogeology.usgs.gov/7.1.0/Application/presentation/PrinterFriendly/cam2map/cam2map.html).
