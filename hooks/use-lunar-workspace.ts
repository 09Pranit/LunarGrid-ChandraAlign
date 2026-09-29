'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  DEMO_IMAGE,
  DEMO_METRICS,
  DEMO_POINTS,
  LOG_MESSAGES,
  PIPELINE,
  downloadFile,
  tiePointCSV,
  type TiePoint,
} from '@/lib/lunar-data';
import {
  inspectImage,
  DEFAULT_TILE,
  BROWSER_UPLOAD_LIMIT,
  LARGE_RASTER_HELP,
  type Inspection,
  type MetadataMode,
  type Side,
  type Tile,
} from '@/lib/ingestion';
import type { ViewMode } from '@/components/lunar-viewer';
import {
  registerJob,
  downloadArtifact,
  type RegistrationResult,
} from '@/lib/registration-api';
type Fields = {
  sensor: string;
  gsd: string;
  sun: string;
  incidence: string;
  emission: string;
};
type Dataset = {
  name: string;
  file: File | null;
  preview: string;
  metadata: Fields;
  label: string;
  labelFile?: File;
  provenanceFile?: File;
  metadataMode: MetadataMode;
  tile: Tile;
  inspection?: Inspection;
  inspecting?: boolean;
  inspectionError?: string;
};
type Result = RegistrationResult & {
  aligned_preview: string;
  api_base: string;
};
const empty: Fields = {
  sensor: 'Not supplied',
  gsd: 'Not supplied',
  sun: 'Not supplied',
  incidence: 'Not supplied',
  emission: 'Not supplied',
};
const emptyDataset = (): Dataset => ({
  name: 'No image selected',
  file: null,
  preview: '',
  metadata: { ...empty },
  label: '',
  metadataMode: 'auto',
  tile: { ...DEFAULT_TILE },
});
const sampleSource: Dataset = {
  ...emptyDataset(),
  name: 'OHRC_SOUTH_POLE_01.tif',
  file: null,
  preview: DEMO_IMAGE,
  metadata: {
    sensor: 'OHRC',
    gsd: '0.25 m',
    sun: '67.4°',
    incidence: '74.2°',
    emission: '2.8°',
  },
  label: 'Example PDS4 metadata',
};
const sampleReference: Dataset = {
  ...emptyDataset(),
  name: 'LROC_NAC_REFERENCE.tif',
  file: null,
  preview: DEMO_IMAGE,
  metadata: {
    sensor: 'LROC NAC',
    gsd: '2.0 m',
    sun: '72.1°',
    incidence: '69.3°',
    emission: '1.2°',
  },
  label: 'Example PDS4 metadata',
};
export function useLunarWorkspace() {
  const [source, setSource] = useState<Dataset>(emptyDataset),
    [reference, setReference] = useState<Dataset>(emptyDataset);
  const [demoLoaded, setDemoLoaded] = useState(false);
  const [confirmOverlap, setConfirmOverlap] = useState(false);
  const inspectEpoch = useRef({ source: 0, reference: 0 }),
    inspectRequest = useRef<{
      source: AbortController | null;
      reference: AbortController | null;
    }>({ source: null, reference: null });
  const [stage, setStage] = useState<
      'ready' | 'running' | 'complete' | 'review_required'
    >('ready'),
    [progress, setProgress] = useState(0),
    [activeIndex, setActiveIndex] = useState(-1);
  const [mode, setMode] = useState<ViewMode>('comparison'),
    [selected, setSelected] = useState<TiePoint | null>(null),
    [showRejected, setShowRejected] = useState(true);
  const [result, setResult] = useState<Result | null>(null),
    [error, setError] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [pipelineOpen, setPipelineOpen] = useState(false),
    [logOpen, setLogOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false),
    [exportNotice, setExportNotice] = useState(false);
  const [endpoint, setEndpoint] = useState(
      process.env.NEXT_PUBLIC_API_URL || '',
    ),
    [draftEndpoint, setDraftEndpoint] = useState(endpoint);
  const [labelTarget, setLabelTarget] = useState<'source' | 'reference'>(
    'source',
  );
  const runId = useRef(0),
    request = useRef<AbortController | null>(null),
    urls = useRef<string[]>([]);
  const uploaded = !!(source.file || reference.file),
    simulated = demoLoaded && !result,
    completed = stage === 'complete' || stage === 'review_required';
  const metrics = result?.metrics || DEMO_METRICS,
    points = completed
      ? result?.tie_points || (uploaded ? [] : DEMO_POINTS)
      : [];
  const sourcePreview = result?.source_preview || source.preview,
    referencePreview = result?.reference_preview || reference.preview;
  useEffect(
    () => () => {
      runId.current++;
      request.current?.abort();
      inspectRequest.current.source?.abort();
      inspectRequest.current.reference?.abort();
      urls.current.forEach((url) => URL.revokeObjectURL(url));
    },
    [],
  );
  const log = useCallback(
    (line: string) =>
      setLogs((l) => [
        ...l,
        `[${new Date().toLocaleTimeString('en-GB', { hour12: false })}] ${line}`,
      ]),
    [],
  );
  const invalidate = () => {
    runId.current++;
    request.current?.abort();
    setStage('ready');
    setProgress(0);
    setActiveIndex(-1);
    setResult(null);
    setSelected(null);
    setError('');
    setMode('comparison');
    setConfirmOverlap(false);
    setExportNotice(false);
  };
  const setter = (target: Side) =>
    target === 'source' ? setSource : setReference;
  const cancelInspection = (target: Side) => {
    inspectEpoch.current[target]++;
    inspectRequest.current[target]?.abort();
  };
  const releasePreviews = () => {
    urls.current.forEach((url) => URL.revokeObjectURL(url));
    urls.current = [];
  };
  const resetInput = (target: Side, change: Partial<Dataset>) => {
    invalidate();
    cancelInspection(target);
    if (demoLoaded) {
      setDemoLoaded(false);
      setSource(emptyDataset());
      setReference(emptyDataset());
    }
    setter(target)((d) => ({
      ...d,
      ...change,
      metadata: { ...empty },
      inspection: undefined,
      inspecting: false,
      inspectionError: '',
      preview: '',
    }));
  };
  const loadImage = (file: File | undefined, target: Side) => {
    if (!file) return;
    invalidate();
    cancelInspection(target);
    const old = target === 'source' ? source : reference;
    if (old.preview.startsWith('blob:')) URL.revokeObjectURL(old.preview);
    if (demoLoaded) {
      cancelInspection(target === 'source' ? 'reference' : 'source');
      (target === 'source' ? setReference : setSource)(emptyDataset());
    }
    setDemoLoaded(false);
    if (file.size > BROWSER_UPLOAD_LIMIT) {
      setter(target)({
        ...emptyDataset(),
        name: file.name,
        inspectionError: LARGE_RASTER_HELP,
      });
      setError(LARGE_RASTER_HELP);
      return;
    }
    const preview = /\.(png|jpe?g|webp|bmp)$/i.test(file.name)
      ? URL.createObjectURL(file)
      : '';
    if (preview) urls.current.push(preview);
    setter(target)({ ...emptyDataset(), name: file.name, file, preview });
    log(`Loaded ${target}: ${file.name} · awaiting server inspection`);
  };
  const loadMetadata = (file: File | undefined, target: Side = labelTarget) => {
    if (!file) return;
    resetInput(target, {
      label: file.name,
      labelFile: file,
      provenanceFile: undefined,
    });
    if (file.size > 1024 * 1024)
      setter(target)((d) => ({
        ...d,
        inspectionError: 'PDS4 XML must be at most 1 MiB.',
      }));
  };
  const loadProvenance = (file: File | undefined, target: Side) => {
    if (!file) return;
    resetInput(target, {
      provenanceFile: file,
      label: '',
      labelFile: undefined,
    });
    if (file.size > 2 * 1024 * 1024)
      setter(target)((d) => ({
        ...d,
        inspectionError: 'Tile provenance must be at most 2 MiB.',
      }));
  };
  const setMetadataMode = (target: Side, metadataMode: MetadataMode) =>
    resetInput(target, { metadataMode });
  const clearLabel = (target: Side) =>
    resetInput(target, { label: '', labelFile: undefined });
  const clearProvenance = (target: Side) =>
    resetInput(target, { provenanceFile: undefined });
  const setTile = (target: Side, tile: Tile) => {
    invalidate();
    cancelInspection(target);
    if (demoLoaded) {
      setDemoLoaded(false);
      setSource(emptyDataset());
      setReference(emptyDataset());
    }
    setter(target)((d) => ({
      ...d,
      tile,
      preview: '',
      inspecting: false,
      inspectionError: '',
      inspection: d.inspection
        ? { ...d.inspection, tile: null, preview: null }
        : undefined,
    }));
  };
  const inspect = async (target: Side) => {
    const data = target === 'source' ? source : reference;
    if (!data.file) return;
    invalidate();
    cancelInspection(target);
    const epoch = inspectEpoch.current[target];
    const controller = new AbortController();
    inspectRequest.current[target] = controller;
    setter(target)((d) => ({
      ...d,
      inspecting: true,
      inspectionError: '',
      preview: '',
    }));
    const timeout = setTimeout(
      () =>
        controller.abort(
          new Error('Inspection timed out. Retry a smaller local tile.'),
        ),
      120000,
    );
    try {
      if (!endpoint)
        throw Error(
          'Connect the processing service in Settings before inspecting files.',
        );
      if (
        (data.labelFile?.size || 0) > 1024 * 1024 ||
        (data.provenanceFile?.size || 0) > 2 * 1024 * 1024
      )
        throw Error('Label/provenance exceeds its size limit.');
      const info = await inspectImage(
        endpoint,
        data.file,
        data.metadataMode,
        data.labelFile,
        data.provenanceFile,
        data.inspection ? data.tile : null,
        controller.signal,
      );
      if (epoch !== inspectEpoch.current[target]) return;
      const m = info.metadata,
        format = (v: number | null, unit: string) =>
          v === null ? 'Not supplied' : `${v} ${unit}`;
      setter(target)((d) => ({
        ...d,
        inspection: info,
        inspecting: false,
        preview: info.preview || '',
        tile: info.tile || {
          x: 0,
          y: 0,
          width: Math.min(512, m.layout.width),
          height: Math.min(512, m.layout.height),
          band: 0,
        },
        metadata: {
          sensor: m.instrument || 'Not supplied',
          gsd: format(m.gsd_meters, 'm'),
          sun: format(m.solar_azimuth_deg, '°'),
          incidence: format(m.incidence_angle_deg, '°'),
          emission: format(m.emission_angle_deg, '°'),
        },
      }));
      log(
        `${target}: ${info.state} · ${info.tile ? 'tile validated' : 'select and validate a tile'}`,
      );
    } catch (e) {
      if (epoch !== inspectEpoch.current[target]) return;
      const message = e instanceof Error ? e.message : 'Inspection failed';
      setter(target)({
        ...data,
        inspecting: false,
        inspectionError: message,
        inspection: undefined,
        preview: '',
        metadata: { ...empty },
      });
      setError(message);
    } finally {
      clearTimeout(timeout);
    }
  };
  const run = useCallback(async () => {
    if (stage === 'running') return;
    if (!uploaded && !demoLoaded) {
      setError('Load both images or choose Load demo data.');
      return;
    }
    if (uploaded && (!source.file || !reference.file)) {
      setError('Load both source and reference images to process your pair.');
      return;
    }
    if (
      uploaded &&
      (!source.inspection?.tile ||
        !reference.inspection?.tile ||
        source.inspecting ||
        reference.inspecting)
    ) {
      setError(
        'Inspect and validate a tile on each image card before running.',
      );
      return;
    }
    if (
      uploaded &&
      (!source.inspection?.metadata.grid ||
        !reference.inspection?.metadata.grid) &&
      !confirmOverlap
    ) {
      setError(
        'Confirm corresponding tiles for image-only matching: overlap is unknown.',
      );
      return;
    }
    if (uploaded && !endpoint) {
      setError(
        'Connect your processing service in Settings to register uploaded images. The sample pair runs locally as a simulation.',
      );
      return;
    }
    const id = ++runId.current;
    setError('');
    setResult(null);
    setSelected(null);
    setStage('running');
    setProgress(0);
    setActiveIndex(0);
    setPipelineOpen(true);
    setLogs([]);
    setMode('comparison');
    if (uploaded) {
      log('Sending image pair to processing service');
      setProgress(15);
      setActiveIndex(1);
      const controller = new AbortController();
      request.current = controller;
      const timeout = setTimeout(
        () =>
          controller.abort(
            new Error('Registration timed out after 15 minutes.'),
          ),
        900000,
      );
      try {
        const form = new FormData();
        form.append('source_file', source.file!);
        form.append('reference_file', reference.file!);
        for (const [side, data] of [
          ['source', source],
          ['reference', reference],
        ] as const) {
          if (data.labelFile) form.append(`${side}_label`, data.labelFile);
          if (data.provenanceFile)
            form.append(`${side}_provenance`, data.provenanceFile);
        }
        form.append(
          'params',
          JSON.stringify({
            schema_version: 2,
            source_mode: source.metadataMode,
            reference_mode: reference.metadataMode,
            source_tile: source.tile,
            reference_tile: reference.tile,
            source_sha256: source.inspection?.metadata.sha256,
            reference_sha256: reference.inspection?.metadata.sha256,
            confirm_unknown_overlap: confirmOverlap,
          }),
        );
        const base = endpoint.replace(/\/$/, '');
        const data = await registerJob(
          base,
          form,
          controller.signal,
          (percent, pipelineStage) => {
            if (id !== runId.current) return;
            setProgress(percent);
            const stages = [
              'ingestion',
              'routing',
              'conditioning',
              'matching',
              'filtering',
              'tps_alignment',
              'export',
            ];
            setActiveIndex(Math.max(0, stages.indexOf(pipelineStage)));
          },
        );
        if (id !== runId.current) return;
        setResult({
          ...data,
          aligned_preview: data.artifacts.registered_preview_base64 || '',
          api_base: base,
        });
        setStage(data.status);
        setProgress(100);
        setActiveIndex(7);
        log(
          `${data.status === 'review_required' ? 'Review required' : 'Registration complete'} · ${data.metrics.active_engine}`,
        );
        if (data.status === 'review_required')
          setError(
            `Review required: ${(data.review_reasons || []).join('; ')}`,
          );
        log(
          'Quality is measured on correspondences withheld from the TPS fit.',
        );
      } catch (e) {
        if (id !== runId.current) return;
        setStage('ready');
        setProgress(0);
        setActiveIndex(-1);
        const message = e instanceof Error ? e.message : 'Registration failed';
        setError(message);
        log(`Failed: ${message}`);
      } finally {
        clearTimeout(timeout);
      }
      return;
    }
    log('SIMULATED PROTOTYPE RUN · example metadata and correspondences');
    for (let i = 0; i < PIPELINE.length; i++) {
      if (id !== runId.current) return;
      setActiveIndex(i);
      log(LOG_MESSAGES[i]);
      await new Promise((r) =>
        setTimeout(r, [400, 430, 500, 570, 420, 470, 390][i]),
      );
      if (id !== runId.current) return;
      setProgress(Math.round(((i + 1) / 7) * 100));
    }
    setActiveIndex(7);
    setStage('complete');
    log('Registration complete · simulated results');
  }, [
    stage,
    uploaded,
    demoLoaded,
    source,
    reference,
    endpoint,
    confirmOverlap,
    log,
  ]);
  const loadDemo = () => {
    invalidate();
    cancelInspection('source');
    cancelInspection('reference');
    releasePreviews();
    setDemoLoaded(true);
    setSource(sampleSource);
    setReference(sampleReference);
    setLogs([
      '[DEMO] Example pair loaded. Run Registration to start the simulation.',
    ]);
  };
  const clearWorkspace = () => {
    invalidate();
    cancelInspection('source');
    cancelInspection('reference');
    releasePreviews();
    setDemoLoaded(false);
    setSource(emptyDataset());
    setReference(emptyDataset());
    setLogs([]);
  };
  const demoRun = useRef(run);
  useEffect(() => {
    demoRun.current = async () => {
      if (!demoLoaded)
        throw Error('Load demo data before running the demonstration tool.');
      await run();
    };
  }, [demoLoaded, run]);
  useEffect(() => {
    const context = (
      document as Document & {
        modelContext?: {
          registerTool: (
            tool: unknown,
            options?: { signal?: AbortSignal },
          ) => void | Promise<void>;
        };
      }
    ).modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    void Promise.resolve(
      context.registerTool(
        {
          name: 'run_lunar_coregistration_demo',
          title: 'Run lunar co-registration demo',
          description:
            'Replay the visible simulated registration pipeline. Requires the sample pair.',
          inputSchema: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: false },
          execute: async () => {
            await demoRun.current();
            return {
              mode: 'demonstration',
              notice: 'SIMULATED PROTOTYPE RESULTS',
            };
          },
        },
        { signal: lifecycle.signal },
      ),
    ).catch(() => {});
    return () => lifecycle.abort();
  }, []);
  const exportReport = () => {
    if (!completed) return;
    if (result) {
      downloadFile(
        JSON.stringify(result, null, 2),
        'lunargrid-registration-report.json',
        'application/json',
      );
      return;
    }
    const report = {
      title: 'LunarGrid / Chandra-Align Registration Report',
      result_type: simulated
        ? 'SIMULATED PROTOTYPE RESULTS'
        : 'BACKEND RESULTS',
      created_at: new Date().toISOString(),
      source_dataset: source.name,
      reference_dataset: reference.name,
      source_metadata: source.metadata,
      reference_metadata: reference.metadata,
      source_pds4_label: source.label || null,
      reference_pds4_label: reference.label || null,
      registration_method: metrics.registration_method || metrics.engine,
      matching_engine: metrics.engine,
      accepted_matches: metrics.accepted_matches,
      rejected_matches: metrics.candidate_matches - metrics.accepted_matches,
      inlier_ratio: metrics.inlier_ratio,
      rmse_px: metrics.rmse_px,
      spatial_coverage: metrics.spatial_coverage,
      runtime_seconds: metrics.runtime_seconds,
      tie_point_count: points.length,
      coordinate_system:
        'Raster pixels; top-left origin; no lunar georeferencing asserted',
      validation_note: simulated
        ? 'Simulated values, not experimentally achieved. Dataset names and acquisition metadata are illustrative. Photograph: Tycho crater, LROC WAC, NASA/GSFC/Arizona State University.'
        : metrics.validation_note,
    };
    downloadFile(
      JSON.stringify(report, null, 2),
      `lunargrid-${simulated ? 'SIMULATED-' : ''}registration-report.json`,
      'application/json',
    );
    log('Exported registration report');
  };
  const exportCSV = () => {
    if (result) {
      if (!result.artifacts.tie_points_csv_url) {
        setError(
          'No tie-point export: this pair failed the quality gates. The report includes the failure reasons.',
        );
        return;
      }
      downloadArtifact(result.api_base, result.artifacts.tie_points_csv_url);
    } else {
      downloadFile(
        tiePointCSV(points, simulated),
        `lunargrid-${simulated ? 'SIMULATED-' : ''}tie-points.csv`,
        'text/csv',
      );
    }
    log('Exported tie points CSV');
  };
  const exportGeoTiff = () => {
    if (!result?.artifacts.geotiff_download_url) {
      setExportNotice(true);
      return;
    }
    downloadArtifact(result.api_base, result.artifacts.geotiff_download_url);
    log(
      result.artifacts.georeferenced
        ? 'Exported registered Moon 2000 GeoTIFF'
        : 'Exported registered TIFF in pixel coordinates; lunar reference grid was not supplied',
    );
  };
  const saveSettings = () => {
    try {
      if (draftEndpoint) {
        const url = new URL(draftEndpoint);
        if (
          !['http:', 'https:'].includes(url.protocol) ||
          url.username ||
          url.password ||
          url.search ||
          url.hash
        )
          throw Error();
      }
      invalidate();
      cancelInspection('source');
      cancelInspection('reference');
      setSource((d) => ({
        ...d,
        inspection: undefined,
        preview: '',
        metadata: { ...empty },
        inspecting: false,
      }));
      setReference((d) => ({
        ...d,
        inspection: undefined,
        preview: '',
        metadata: { ...empty },
        inspecting: false,
      }));
      setEndpoint(draftEndpoint.replace(/\/$/, ''));
      setSettingsOpen(false);
      setError('');
    } catch {
      setError(
        'Use an HTTP or HTTPS service URL without embedded credentials or query parameters.',
      );
      setSettingsOpen(false);
    }
  };
  const meters = (s: string) => {
    const match = s.match(/^([\d.]+)\s*(m|km|cm)(?:\s*\/\s*(?:px|pixel))?$/i);
    return match
      ? Number(match[1]) *
          ({ m: 1, km: 1000, cm: 0.01 }[match[2].toLowerCase()] || 1)
      : NaN;
  };
  const a = meters(source.metadata.gsd),
    b = meters(reference.metadata.gsd),
    scaleGap =
      a > 0 && b > 0 ? `${(Math.max(a, b) / Math.min(a, b)).toFixed(1)}×` : '—';
  return {
    source,
    reference,
    stage,
    progress,
    activeIndex,
    mode,
    setMode,
    selected,
    setSelected,
    showRejected,
    setShowRejected,
    result,
    error,
    logs,
    pipelineOpen,
    setPipelineOpen,
    logOpen,
    setLogOpen,
    settingsOpen,
    setSettingsOpen,
    exportNotice,
    setExportNotice,
    endpoint,
    draftEndpoint,
    setDraftEndpoint,
    labelTarget,
    setLabelTarget,
    demoLoaded,
    uploaded,
    simulated,
    completed,
    metrics,
    points,
    sourcePreview,
    referencePreview,
    loadImage,
    loadMetadata,
    loadProvenance,
    setMetadataMode,
    clearLabel,
    clearProvenance,
    setTile,
    inspect,
    confirmOverlap,
    setConfirmOverlap,
    run,
    loadDemo,
    clearWorkspace,
    exportReport,
    exportCSV,
    exportGeoTiff,
    saveSettings,
    scaleGap,
  };
}
