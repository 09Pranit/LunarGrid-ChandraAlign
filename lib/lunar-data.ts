export type TiePoint = { id: string; source_x: number; source_y: number; reference_x: number; reference_y: number; confidence: number | null; status: 'accepted' | 'rejected' };
export const DEMO_IMAGE = '/lunar-tycho.jpg';
export const DEMO_POINTS: TiePoint[] = Array.from({ length: 48 }, (_, i) => {
  const x = 100 + (i % 8) * 151 + ((i * 29) % 47);
  const y = 93 + Math.floor(i / 8) * 206 + ((i * 17) % 53);
  const rejected = i % 8 === 5;
  return { id: `LG-${String(i + 1).padStart(4, '0')}`, source_x: x + 18, source_y: y - 12, reference_x: x + (rejected ? 61 : 0), reference_y: y + (rejected ? -43 : 0), confidence: rejected ? 0.36 + (i % 5) * 0.04 : 0.91 + (i % 8) * 0.01, status: rejected ? 'rejected' : 'accepted' };
});
export const DEMO_METRICS = { candidate_matches: 48, accepted_matches: 42, rejected_matches: 6, inlier_ratio: 0.875, rmse_px: 0.42, spatial_coverage: 0.86, runtime_seconds: 3.18, engine: 'Auto → LightGlue', registration_method: 'TPS (simulated)', validation_note: 'Illustrative values, not experimentally achieved results.' };
export const PIPELINE = ['PDS4 Metadata Read', 'Scale / Image Conditioning', 'Feature Detection', 'Feature Matching', 'Outlier Removal', 'TPS Alignment', 'RMSE Validation'];
export const LOG_MESSAGES = ['Reading PDS4 metadata', 'Preparing image pair', 'Detecting features', 'Matching features', 'Removing outliers', 'Applying TPS alignment', 'Computing RMSE'];
export function downloadFile(content: string | Blob, filename: string, type = 'text/plain') {
  const url = URL.createObjectURL(content instanceof Blob ? content : new Blob([content], { type }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function tiePointCSV(points: TiePoint[], simulated: boolean) {
  const rows = [['mode','match_id','source_x_px','source_y_px','reference_x_px','reference_y_px','confidence','status'], ...points.map(p => [simulated ? 'SIMULATED PROTOTYPE RESULTS' : 'BACKEND RESULTS', p.id, p.source_x, p.source_y, p.reference_x, p.reference_y, p.confidence ?? '', p.status])];
  return rows.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n');
}
