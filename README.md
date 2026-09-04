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
