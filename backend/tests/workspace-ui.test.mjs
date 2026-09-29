// Run against `pnpm dev`: node --test backend/tests/workspace-ui.test.mjs
// Install playwright 1.62.1 + Chromium, or set LUNARGRID_PLAYWRIGHT_MODULE to its module URL.
import assert from 'node:assert/strict';
import { test } from 'node:test';
const { chromium } = await import(
  process.env.LUNARGRID_PLAYWRIGHT_MODULE || 'playwright'
);
const base = process.env.LUNARGRID_UI_URL || 'http://127.0.0.1:3000';
const png =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
const id = 'job_lunar_' + 'b'.repeat(32);
const meta = (mode = 'auto') => ({
  schema_version: 2,
  source_format: mode === 'none' ? 'IMAGE_ONLY' : 'PDS3',
  label_location: mode === 'none' ? 'none' : 'attached',
  selected_mode: mode,
  file_name: 'test.img',
  file_size: 100,
  sha256: 'a'.repeat(64),
  product_id: 'TEST_PRODUCT',
  processing_level: 'CDR',
  instrument: null,
  gsd_meters: null,
  incidence_angle_deg: null,
  emission_angle_deg: null,
  solar_azimuth_deg: null,
  geometry_status: 'unknown',
  missing_fields: ['gsd_meters', 'grid'],
  warnings: ['No verified ground transform.'],
  layout: {
    width: 128,
    height: 128,
    bands: 1,
    dtype: '<i2',
    sample_type: 'LSB_INTEGER',
  },
  grid: null,
  lineage: {},
});

test('independent metadata modes, accessible errors, tile validation, stale results and demo isolation', async () => {
  const browser = await chromium.launch({
    headless: true,
    channel: process.env.LUNARGRID_BROWSER_CHANNEL || undefined,
  });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    let submitted;
    await page.route('http://service.test/**', async (route) => {
      const url = route.request().url();
      const body = route.request().postDataBuffer()?.toString() || '';
      if (url.endsWith('/inspect')) {
        const mode = /name="mode"\r\n\r\n([^\r]+)/.exec(body)?.[1] || 'auto';
        const tile = JSON.parse(
          /name="tile"\r\n\r\n([^\r]+)/.exec(body)?.[1] || 'null',
        );
        if (body.includes('MALFORMED_XML') && mode !== 'pds3') {
          return route.fulfill({
            status: 422,
            json: {
              detail:
                'Supplied XML is invalid. Correct the XML or explicitly select Embedded PDS3.',
            },
          });
        }
        const metadata = meta(mode);
        if (body.includes('MALFORMED_XML'))
          metadata.warnings.push(
            'Explicit PDS3 override: supplied XML is invalid.',
          );
        return route.fulfill({
          json: {
            metadata,
            state: mode === 'none' ? 'No usable label' : 'PDS3 attached label',
            tile,
            preview: tile ? png : null,
            validation: 'server-validated uploaded bytes',
            max_tile_pixels: 1048576,
          },
        });
      }
      if (url.endsWith('/jobs')) {
        submitted = body;
        return route.fulfill({ json: { job_id: id, status: 'queued' } });
      }
      if (url.endsWith('/results'))
        return route.fulfill({
          json: {
            job_id: id,
            status: 'review_required',
            metrics: {
              rmse_px: 0.2,
              rmse_basis: 'withheld_feature_correspondences',
              vsui_score: 0.9,
              active_engine: 'sift_flann',
              engine: 'sift_flann',
              accepted_matches: 50,
              candidate_matches: 100,
              inlier_ratio: 0.9,
              spatial_coverage: 0.8,
              runtime_seconds: 1,
            },
            artifacts: {
              registered_preview_base64: png,
              geotiff_download_url: `/api/v1/artifacts/${id}.tif`,
              tie_points_csv_url: `/api/v1/artifacts/${id}_tiepoints.csv`,
              georeferenced: false,
            },
            metadata: { source: meta(), reference: meta('none') },
            tiles: {},
            matching_mode: 'image_only',
            overlap_status: 'unknown',
            review_reasons: ['Overlap unknown; visual review required.'],
          },
        });
      return route.fulfill({
        json: { job_id: id, status: 'review_required', progress_percent: 100 },
      });
    });
    await page.goto(base);
    await page
      .getByRole('button', { name: 'Load demo data', exact: true })
      .waitFor();
    assert.equal(
      await page
        .getByRole('combobox', { name: 'Source metadata source' })
        .inputValue(),
      'auto',
    );
    assert.equal(
      await page
        .getByRole('combobox', { name: 'Reference metadata source' })
        .inputValue(),
      'auto',
    );
    assert.equal(
      await page.getByText('OHRC_SOUTH_POLE_01.tif', { exact: true }).count(),
      0,
    );
    await page
      .getByRole('button', { name: 'Load demo data', exact: true })
      .click();
    await page.getByText('OHRC_SOUTH_POLE_01.tif', { exact: true }).waitFor();
    await page
      .getByRole('button', { name: 'Clear workspace', exact: true })
      .click();
    assert.equal(
      await page.getByText('OHRC_SOUTH_POLE_01.tif', { exact: true }).count(),
      0,
    );
    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await page.getByLabel('Service URL').fill('http://service.test');
    await page
      .getByRole('button', { name: 'Save connection', exact: true })
      .click();
    for (const side of ['source', 'reference'])
      await page
        .getByLabel(`Load ${side} image`, { exact: true })
        .setInputFiles({
          name: `${side}.img`,
          mimeType: 'application/octet-stream',
          buffer: Buffer.from('test bytes'),
        });
    assert.equal(
      await page
        .getByText(
          'Not inspected · browser preview is not authoritative metadata',
          { exact: true },
        )
        .count(),
      2,
    );
    await page
      .getByRole('combobox', { name: 'Reference metadata source' })
      .selectOption('none');
    await page
      .getByLabel('Load source PDS4 label')
      .setInputFiles({
        name: 'bad.xml',
        mimeType: 'application/xml',
        buffer: Buffer.from('MALFORMED_XML'),
      });
    const source = page.getByRole('region', { name: 'Source image input' });
    const reference = page.getByRole('region', {
      name: 'Reference image input',
    });
    await source
      .getByRole('button', { name: 'Inspect image and label' })
      .click();
    await source
      .getByRole('alert')
      .filter({ hasText: 'Supplied XML is invalid' })
      .waitFor();
    // Native select is keyboard accessible; the override does not change the other side.
    const select = page.getByRole('combobox', {
      name: 'Source metadata source',
    });
    await select.focus();
    await select.selectOption('pds3');
    assert.equal(
      await page
        .getByRole('combobox', { name: 'Reference metadata source' })
        .inputValue(),
      'none',
    );
    await source
      .getByRole('button', { name: 'Inspect image and label' })
      .click();
    await source
      .getByText('PDS3 attached label · server validated', { exact: true })
      .waitFor();
    await source
      .getByText('Explicit PDS3 override: supplied XML is invalid.')
      .waitFor();
    await page.getByLabel('Source tile x', { exact: true }).fill('9');
    await page.getByLabel('Source tile y', { exact: true }).fill('11');
    await page.getByLabel('Source tile width', { exact: true }).fill('64');
    await page.getByLabel('Source tile height', { exact: true }).fill('64');
    await source
      .getByText('Full-image origin: (9, 11) · 64 × 64 pixels', { exact: true })
      .waitFor();
    await source
      .getByRole('button', { name: 'Validate selected tile' })
      .click();
    await source
      .getByText(
        'Selected tile validated. Changing inputs invalidates this selection.',
      )
      .waitFor();
    await reference
      .getByRole('button', { name: 'Inspect image and label' })
      .click();
    await reference
      .getByRole('button', { name: 'Validate selected tile' })
      .click();
    await reference
      .getByText(
        'Selected tile validated. Changing inputs invalidates this selection.',
      )
      .waitFor();
    await page
      .getByRole('checkbox', { name: /I selected corresponding tiles/ })
      .check();
    await page
      .getByRole('button', { name: 'Run Registration', exact: true })
      .click();
    await page.getByText('BACKEND RESULTS', { exact: true }).waitFor();
    assert.ok(
      submitted.includes('"source_mode":"pds3"') &&
        submitted.includes('"reference_mode":"none"'),
    );
    assert.ok(submitted.includes('"x":9,"y":11'));
    await page.getByLabel('Source tile x', { exact: true }).fill('10');
    await page.getByText('AWAITING REGISTRATION', { exact: true }).waitFor();
    assert.equal(
      await page.getByText('BACKEND RESULTS', { exact: true }).count(),
      0,
    );
    await page
      .getByRole('button', { name: 'Load demo data', exact: true })
      .click();
    await page
      .getByLabel('Load source image', { exact: true })
      .setInputFiles({
        name: 'real.img',
        mimeType: 'application/octet-stream',
        buffer: Buffer.from('real'),
      });
    assert.equal(
      await page.getByText('LROC_NAC_REFERENCE.tif', { exact: true }).count(),
      0,
    );
    assert.equal(
      await page.getByText('OHRC_SOUTH_POLE_01.tif', { exact: true }).count(),
      0,
    );
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
});
