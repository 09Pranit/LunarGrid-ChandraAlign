import assert from 'node:assert/strict';
import { test } from 'node:test';
import { registerJob } from '../../lib/registration-api.ts';

const id = 'job_lunar_' + 'a'.repeat(32);
const result = {
  job_id: id,
  status: 'complete',
  metrics: {
    rmse_px: 0.3,
    vsui_score: 0.8,
    active_engine: 'sift_flann',
    accepted_matches: 32,
    candidate_matches: 100,
    inlier_ratio: 0.9,
    spatial_coverage: 0.8,
    runtime_seconds: 2,
  },
  artifacts: {
    registered_preview_base64: 'data:image/png;base64,iVBORw0KGgo=',
    geotiff_download_url: `/api/v1/artifacts/${id}.tif`,
    tie_points_csv_url: `/api/v1/artifacts/${id}_tiepoints.csv`,
  },
};

test('submits versioned multipart and polls queued through complete', async (t) => {
  const replies = [
    { job_id: id, status: 'queued' },
    { status: 'queued', progress_percent: 0 },
    { status: 'processing', stage: 'matching', progress_percent: 40 },
    { status: 'complete', progress_percent: 100 },
    result,
  ];
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    calls.push([url, options]);
    return Response.json(replies.shift());
  });
  const form = new FormData();
  form.append('source_file', new Blob(['source']), 'source.png');
  form.append('reference_file', new Blob(['reference']), 'reference.png');
  const progress = [];
  const data = await registerJob(
    'http://localhost:8000',
    form,
    new AbortController().signal,
    (n) => progress.push(n),
  );
  assert.deepEqual(data, result);
  assert.deepEqual(progress, [0, 40, 100]);
  assert.equal(calls[0][0], 'http://localhost:8000/api/v1/registration/jobs');
  assert.equal(calls[0][1].body, form);
  assert.equal(
    calls.at(-1)[0],
    `http://localhost:8000/api/v1/registration/jobs/${id}/results`,
  );
});

test('failed jobs and malformed review responses surface clear messages', async (t) => {
  const replies = [
    { job_id: id },
    { status: 'failed', progress_percent: 100, error: 'Worker failed' },
    { job_id: id },
    { status: 'review_required', progress_percent: 100 },
    {
      job_id: id,
      status: 'review_required',
      artifacts: {},
      review_reasons: ['VSUI too low'],
    },
  ];
  t.mock.method(globalThis, 'fetch', async () =>
    Response.json(replies.shift()),
  );
  await assert.rejects(
    registerJob('', new FormData(), new AbortController().signal, () => {}),
    /Worker failed/,
  );
  await assert.rejects(
    registerJob('', new FormData(), new AbortController().signal, () => {}),
    /missing quality metrics/,
  );
});

test('cancellation stops polling and quality gate rejects false completion', async (t) => {
  const controller = new AbortController();
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () =>
    Response.json(
      ++requests === 1
        ? { job_id: id }
        : { status: 'queued', progress_percent: 0 },
    ),
  );
  await assert.rejects(
    registerJob('', new FormData(), controller.signal, () =>
      controller.abort(new Error('cancelled')),
    ),
    /cancelled/,
  );
  assert.equal(requests, 3); // includes server cancellation request
  const replies = [
    { job_id: id },
    { status: 'complete', progress_percent: 100 },
    { ...result, metrics: { ...result.metrics, rmse_px: 0.7 } },
  ];
  t.mock.method(globalThis, 'fetch', async () =>
    Response.json(replies.shift()),
  );
  await assert.rejects(
    registerJob('', new FormData(), new AbortController().signal, () => {}),
    /quality thresholds/,
  );
});
