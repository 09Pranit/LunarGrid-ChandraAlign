'use client';
import type { Inspection, MetadataMode, Side, Tile } from '@/lib/ingestion';
import { LARGE_RASTER_HELP } from '@/lib/ingestion';

type Props = {
  target: Side;
  name: string;
  hasFile: boolean;
  demo: boolean;
  mode: MetadataMode;
  tile: Tile;
  label?: string;
  provenanceName?: string;
  inspection?: Inspection;
  busy?: boolean;
  error?: string;
  onImage: (file?: File) => void;
  onLabel: (file?: File) => void;
  onProvenance: (file?: File) => void;
  onMode: (mode: MetadataMode) => void;
  onTile: (tile: Tile) => void;
  onInspect: () => void;
  onClearLabel: () => void;
  onClearProvenance: () => void;
};

export function ImageIngestionCard(p: Props) {
  const meta = p.inspection?.metadata;
  const title = p.target === 'source' ? 'Source' : 'Reference';
  const origin = meta?.lineage.full_image_origin || [0, 0];
  return (
    <section className="ingestion-card" aria-label={`${title} image input`}>
      <label className="upload-card">
        <input
          aria-label={`Load ${p.target} image`}
          type="file"
          accept="image/*,.img"
          onChange={(e) => {
            p.onImage(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
        <span>
          <b>{title} image</b>
          <small>{p.name}</small>
          <em>
            {p.demo
              ? 'DEMO DATA'
              : p.hasFile
                ? 'LOCAL FILE · INSPECT BEFORE RUNNING'
                : 'NO FILE LOADED'}
          </em>
        </span>
      </label>
      <label htmlFor={`${p.target}-metadata-mode`}>Metadata source</label>
      <select
        id={`${p.target}-metadata-mode`}
        aria-label={`${title} metadata source`}
        value={p.mode}
        onChange={(e) => p.onMode(e.target.value as MetadataMode)}
      >
        <option value="auto">Auto</option>
        <option value="pds4">PDS4 XML</option>
        <option value="pds3">Embedded PDS3</option>
        <option value="none">None / image only</option>
      </select>
      <label className="ingestion-file">
        PDS4 XML for {p.target}
        <input
          aria-label={`Load ${p.target} PDS4 label`}
          type="file"
          accept=".xml"
          disabled={!p.hasFile}
          onChange={(e) => {
            p.onLabel(e.target.files?.[0]);
            e.target.value = '';
          }}
        />
      </label>
      {p.label && (
        <p className="file-selection">
          {p.label}{' '}
          <button
            type="button"
            onClick={p.onClearLabel}
            aria-label={`Clear ${p.target} XML`}
          >
            Clear XML
          </button>
        </p>
      )}
      <details>
        <summary>Large scene / local tile provenance</summary>
        <p>{LARGE_RASTER_HELP}</p>
        <label className="ingestion-file">
          Local tile manifest
          <input
            type="file"
            accept=".json"
            aria-label={`Load ${p.target} tile provenance`}
            disabled={!p.hasFile}
            onChange={(e) => {
              p.onProvenance(e.target.files?.[0]);
              e.target.value = '';
            }}
          />
        </label>
        {p.provenanceName && (
          <p>
            {p.provenanceName}{' '}
            <button type="button" onClick={p.onClearProvenance}>
              Clear manifest
            </button>
          </p>
        )}
      </details>
      <output aria-live="polite" className="label-status">
        {p.busy
          ? 'Parsing and validating on server…'
          : p.inspection
            ? `${p.inspection.state} · server validated`
            : p.demo
              ? 'Illustrative demo metadata'
              : 'Not inspected · browser preview is not authoritative metadata'}
      </output>
      {meta && (
        <dl className="label-details">
          <div>
            <dt>Label location</dt>
            <dd>{meta.label_location.replaceAll('_', ' ')}</dd>
          </div>
          <div>
            <dt>Product ID</dt>
            <dd>{meta.product_id || 'Not supplied'}</dd>
          </div>
          <div>
            <dt>Processing level</dt>
            <dd>{meta.processing_level || 'Not supplied'}</dd>
          </div>
          <div>
            <dt>Uploaded raster</dt>
            <dd>
              {meta.layout.width} × {meta.layout.height} · {meta.layout.bands}{' '}
              band(s)
            </dd>
          </div>
          {meta.lineage.original_layout && (
            <div>
              <dt>Original full raster</dt>
              <dd>
                {meta.lineage.original_layout.width} ×{' '}
                {meta.lineage.original_layout.height}
              </dd>
            </div>
          )}
          <div>
            <dt>Decoded type</dt>
            <dd>
              {meta.layout.sample_type} ({meta.layout.dtype})
            </dd>
          </div>
          <div>
            <dt>Geometry</dt>
            <dd>{meta.geometry_status.replaceAll('_', ' ')}</dd>
          </div>
          <div>
            <dt>GSD / incidence</dt>
            <dd>
              {meta.gsd_meters === null ? 'Unknown' : `${meta.gsd_meters} m/px`}{' '}
              /{' '}
              {meta.incidence_angle_deg === null
                ? 'Unknown'
                : `${meta.incidence_angle_deg}°`}
            </dd>
          </div>
          <div>
            <dt>Emission / sun azimuth</dt>
            <dd>
              {meta.emission_angle_deg ?? 'Unknown'} /{' '}
              {meta.solar_azimuth_deg ?? 'Unknown'}
            </dd>
          </div>
          <div>
            <dt>Missing</dt>
            <dd>{meta.missing_fields.join(', ') || 'None'}</dd>
          </div>
        </dl>
      )}
      <fieldset className="tile-fields">
        <legend>Selected tile · zero-based pixels</legend>
        {(['x', 'y', 'width', 'height', 'band'] as const).map((key) => (
          <label key={key}>
            {key}
            <input
              aria-label={`${title} tile ${key}`}
              type="number"
              step="1"
              min={key === 'width' || key === 'height' ? 32 : 0}
              max={key === 'width' || key === 'height' ? 32766 : undefined}
              value={p.tile[key]}
              onChange={(e) =>
                p.onTile({ ...p.tile, [key]: Number(e.target.value) })
              }
            />
          </label>
        ))}
      </fieldset>
      <p className="tile-origin">
        Full-image origin: ({origin[0] + p.tile.x}, {origin[1] + p.tile.y}) ·{' '}
        {p.tile.width} × {p.tile.height} pixels
      </p>
      {meta?.grid ? (
        <p>
          Tile corner footprint:{' '}
          {((((meta.grid.west_lon + p.tile.x * meta.grid.pixel_size_deg + 180) %
            360) +
            360) %
            360) -
            180}
          ° E, {meta.grid.north_lat - p.tile.y * meta.grid.pixel_size_deg}° N;
          extent {p.tile.width * meta.grid.pixel_size_deg}° ×{' '}
          {p.tile.height * meta.grid.pixel_size_deg}°. Moon 2000 geographic
          grid.
        </p>
      ) : (
        <p>
          Footprint / overlap unknown. Select corresponding tiles using
          observation context or external camera geometry.
        </p>
      )}
      {meta?.warnings.map((warning, i) => (
        <p className="ingestion-warning" key={i}>
          {warning}
        </p>
      ))}
      {p.error && (
        <p role="alert" className="error-note">
          {p.error}
        </p>
      )}
      <button
        type="button"
        className="inspect-button"
        disabled={!p.hasFile || p.busy}
        onClick={p.onInspect}
      >
        {p.busy
          ? 'Validating…'
          : p.inspection
            ? 'Validate selected tile'
            : 'Inspect image and label'}
      </button>
      {p.inspection && (
        <p>
          {p.inspection.tile
            ? 'Selected tile validated. Changing inputs invalidates this selection.'
            : 'Check the label details, select a tile above, then validate it.'}
        </p>
      )}
    </section>
  );
}
