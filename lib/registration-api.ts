import type { TiePoint, DEMO_METRICS } from './lunar-data';

export type RegistrationResult = {
  job_id: string;
  status: 'complete' | 'review_required';
  metrics: typeof DEMO_METRICS & { active_engine: string; vsui_score: number; rmse_basis: string };
  artifacts: { registered_preview_base64: string; geotiff_download_url: string; tie_points_csv_url: string; georeferenced: boolean };
  source_preview?: string;
  reference_preview?: string;
  tie_points?: TiePoint[];
  review_reasons?: string[];
  image_dimensions?: { source: { width: number; height: number }; reference: { width: number; height: number } };
};

type JobState = { job_id: string; status: string; progress_percent: number; stage?: string; error?: string };
async function responseJSON<T = JobState>(response: Response): Promise<T> {
  const body: unknown = await response.json();
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw Error('Invalid service response.');
  const data = body as { detail?: string | { message?: string } };
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : data.detail?.message || 'Registration service returned an error.');
  return body as T;
}

function pause(signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) { reject(signal.reason); return; }
    const abort = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, 750);
    signal.addEventListener('abort', abort, { once: true });
  });
}

export async function registerJob(base: string, form: FormData, signal: AbortSignal,
  onProgress: (percent: number, stage: string) => void): Promise<RegistrationResult> {
  const receipt = await responseJSON(await fetch(`${base}/api/v1/registration/jobs`, { method: 'POST', body: form, signal }));
  if (typeof receipt.job_id !== 'string' || !/^job_lunar_[a-f0-9]{32}$/.test(receipt.job_id)) throw Error('The service returned an invalid job ID.');
  for (;;) {
    const state = await responseJSON(await fetch(`${base}/api/v1/registration/jobs/${receipt.job_id}`, { signal, cache: 'no-store' }));
    if (!Number.isFinite(state.progress_percent) || state.progress_percent < 0 || state.progress_percent > 100) throw Error('The service returned invalid progress.');
    onProgress(state.progress_percent, state.stage || state.status);
    if (state.status === 'failed') throw Error(state.error || 'Registration failed.');
    if (state.status === 'complete' || state.status === 'review_required') break;
    if (!['queued', 'processing'].includes(state.status)) throw Error('The service returned an unknown job status.');
    await pause(signal);
  }
  const data = await responseJSON<RegistrationResult>(await fetch(`${base}/api/v1/registration/jobs/${receipt.job_id}/results`, { signal, cache: 'no-store' }));
  if (data.job_id !== receipt.job_id || !['complete', 'review_required'].includes(data.status)) throw Error('The service returned inconsistent results.');
  if (data.status === 'review_required' && !data.artifacts?.registered_preview_base64) throw Error(`Review required: ${(data.review_reasons || ['Insufficient registration quality']).join('; ')}`);
  if (!data.metrics || !data.artifacts?.registered_preview_base64?.startsWith('data:image/png;base64,') ||
      !(['rmse_px','accepted_matches','candidate_matches','inlier_ratio','spatial_coverage','runtime_seconds','vsui_score'] as const).every(k => Number.isFinite(data.metrics[k]))) throw Error('The service response is missing a PNG preview or numeric quality metrics.');
  if (typeof data.metrics.active_engine !== 'string' ||
      data.artifacts.geotiff_download_url !== `/api/v1/artifacts/${data.job_id}.tif` ||
      data.artifacts.tie_points_csv_url !== `/api/v1/artifacts/${data.job_id}_tiepoints.csv`) throw Error('The service returned invalid artifacts.');
  if (data.status === 'complete' && (data.metrics.rmse_px > 0.5 || data.metrics.vsui_score < 0.75)) throw Error('The service marked a result complete despite failing its quality thresholds.');
  if (data.tie_points && !data.tie_points.every(p => typeof p.id === 'string' &&
      [p.source_x,p.source_y,p.reference_x,p.reference_y].every(Number.isFinite) &&
      ['accepted','rejected'].includes(p.status) && (p.confidence === null || Number.isFinite(p.confidence)))) throw Error('The service returned invalid tie-point coordinates.');
  if (data.image_dimensions && ![data.image_dimensions.source, data.image_dimensions.reference].every(s => s && Number.isInteger(s.width) && Number.isInteger(s.height) && s.width > 0 && s.height > 0)) throw Error('The service returned invalid raster dimensions.');
  return data;
}

export function downloadArtifact(base: string, path: string) {
  const anchor = document.createElement('a');
  anchor.href = base + path;
  anchor.download = path.split('/').pop() || 'registration';
  anchor.target = '_blank';
  anchor.rel = 'noopener';
  anchor.click();
}
