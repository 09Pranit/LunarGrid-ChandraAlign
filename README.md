# Chandra-Align

LunarGrid's Smart India Hackathon prototype for multi-sensor lunar image co-registration.

The hosted interface provides a reliable demonstration mode. The local processing service implements Wallis conditioning, SIFT feature extraction, FLANN matching, USAC/MAGSAC geometric verification, warping, spatial-coverage scoring, and JSON validation output.

## Run the interface

```powershell
pnpm dev
```

## Run the processing service

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
.venv\Scripts\uvicorn main:app --app-dir backend --reload --port 8000
```

Open the API documentation at `http://localhost:8000/docs`. Production accuracy claims must use named Chandrayaan/LROC products and independent held-out checkpoints.

## Research workstation controls

The four viewer modes share zoom, pan, fit, and pixel inspection. Pixel values are 8-bit preview samples, not original calibrated DN values. Overlay compares source against reference. Tie Points shows both endpoints, with green circles for accepted matches and red crosses for rejected matches; the selector supports keyboard inspection.

The example uses a single real LROC WAC photograph of Tycho crater (NASA/GSFC/Arizona State University), shifted by 18/-12 pixels for the source. OHRC/NAC dataset names and acquisition metadata are illustrative and do not describe that photograph. All 48 demo correspondences and quality figures are simulated. Reports and CSV rows carry an explicit simulation marker.

Image source: https://science.nasa.gov/image-detail/amf-cc06843a-5b68-4cb4-9119-c62e383b54fa/

For real images, load both files, optionally load each PDS4 XML label, and set the service URL in Settings. Uploading invalidates the simulated results. The existing backend now returns source/reference previews and all accepted/rejected point coordinates in `tie_points`; confidence is null because the classical matcher does not produce calibrated probabilities. Its actual alignment remains homography, and is identified as such in the UI rather than claimed as TPS/LightGlue. Configure `LUNARGRID_ALLOWED_ORIGINS` as a comma-separated list when using another UI origin.

GeoTIFF export requires the backend to supply `registered_geotiff_base64` containing a valid georeferenced lunar raster. The included classical backend does not yet preserve a lunar CRS or emit this field. Until it does, this menu item explains the missing output; it never fabricates georeferencing. CSV and JSON registration reports download directly and include dataset metadata, method, match counts, RMSE, coverage, and runtime.

## Vercel deployment

`vercel.json` builds the same React workstation with `vite.vercel.config.ts`. The `vercel-entry` wrapper reuses `app/page.tsx`, all components, the existing CSS, and image assets. The registration API remains an external service configured in Settings.

Import this GitHub repository into Vercel with these settings:

- Framework Preset: **Vite** (single frontend project).
- Root Directory: **.** (repository root, where `vercel.json` lives).
- Build Command: **npm run build**.
- Output Directory: **vercel-dist**.
- Install Command: leave the automatic default.

If the import screen detects `frontend` and `backend` as multiple services and says "vercel.json required to deploy projects with multiple services", switch to a single Vite project before deploying. Do not accept the generated multi-service configuration: this repository's Vercel configuration deploys the frontend, while `backend/` contains a separately run Python processing service. The public sample workflow runs without that service; real uploaded-image registration requires its URL in Settings.

Build locally with `npm run build`, then deploy with `vercel --prod`. Configure `LUNARGRID_ALLOWED_ORIGINS` on your registration backend to include the final Vercel URL. The original Sites build remains available through `pnpm build:sites`.
