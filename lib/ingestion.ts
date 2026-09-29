export type MetadataMode = 'auto' | 'pds4' | 'pds3' | 'none';
export type Side = 'source' | 'reference';
export type Tile = {
  x: number;
  y: number;
  width: number;
  height: number;
  band: number;
};
export type ImageMetadata = {
  schema_version: 2;
  source_format: 'PDS3' | 'PDS4' | 'IMAGE_ONLY';
  label_location: string;
  selected_mode: MetadataMode;
  file_name: string;
  file_size: number;
  sha256: string;
  product_id: string | null;
  processing_level: string | null;
  instrument: string | null;
  gsd_meters: number | null;
  incidence_angle_deg: number | null;
  emission_angle_deg: number | null;
  solar_azimuth_deg: number | null;
  geometry_status: string;
  missing_fields: string[];
  warnings: string[];
  layout: {
    width: number;
    height: number;
    bands: number;
    dtype: string;
    sample_type: string;
    offset: number;
    prefix_bytes: number;
    suffix_bytes: number;
    scale: number;
    value_offset: number;
  };
  grid: { west_lon: number; north_lat: number; pixel_size_deg: number } | null;
  lineage: {
    full_image_origin?: [number, number];
    original_product_id?: string;
    verification?: string;
    original_layout?: { width: number; height: number };
  };
};
export type Inspection = {
  metadata: ImageMetadata;
  state: string;
  tile: Tile | null;
  preview: string | null;
  validation: string;
  max_tile_pixels: number;
};
export const DEFAULT_TILE: Tile = {
  x: 0,
  y: 0,
  width: 512,
  height: 512,
  band: 0,
};
export const BROWSER_UPLOAD_LIMIT = 512 * 1024 * 1024;
export const LARGE_RASTER_HELP =
  'Browser uploads are limited to 512 MiB per file. Extract a corresponding tile locally with python -m backend.tile_cli --help, then load the TIFF and its .provenance.json on this card.';

export async function inspectImage(
  base: string,
  file: File,
  mode: MetadataMode,
  label: File | undefined,
  provenance: File | undefined,
  tile: Tile | null,
  signal: AbortSignal,
): Promise<Inspection> {
  const form = new FormData();
  form.append('file', file);
  form.append('mode', mode);
  form.append('tile', JSON.stringify(tile));
  if (label) form.append('label', label);
  if (provenance) form.append('provenance', provenance);
  const response = await fetch(`${base}/api/v1/registration/inspect`, {
    method: 'POST',
    body: form,
    signal,
  });
  const data = (await response.json()) as Inspection & { detail?: unknown };
  if (!response.ok)
    throw Error(
      typeof data.detail === 'string'
        ? data.detail
        : 'Image inspection failed.',
    );
  if (
    data?.metadata?.schema_version !== 2 ||
    !['PDS3', 'PDS4', 'IMAGE_ONLY'].includes(data.metadata.source_format) ||
    !/^[a-f0-9]{64}$/.test(data.metadata.sha256) ||
    !Array.isArray(data.metadata.warnings)
  )
    throw Error('Invalid inspection response.');
  return data;
}
