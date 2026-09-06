from __future__ import annotations

import base64
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from registration import RegistrationError, register_images

app = FastAPI(title="Chandra-Align Processing API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("LUNARGRID_ALLOWED_ORIGINS", "http://localhost:3000,https://lunargrid-chandra-align.pranitk0905.chatgpt.site").split(","), allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def decode_image(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def encode_png(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise RegistrationError("The output preview could not be encoded.")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "engine": "classical-core"}


@app.post("/metadata")
async def metadata(label: UploadFile = File(...)) -> dict:
    try:
        root = ET.fromstring(await label.read())
    except ET.ParseError as exc:
        raise HTTPException(400, f"Invalid PDS4 XML: {exc}") from exc
    values = {}
    wanted = ("pixel_resolution", "ground_sampling_distance", "incidence_angle", "emission_angle", "solar_azimuth", "longitude", "latitude")
    for element in root.iter():
        key = element.tag.rsplit("}", 1)[-1].lower()
        if any(term in key for term in wanted) and element.text and element.text.strip():
            values[key] = {"value": element.text.strip(), "unit": element.attrib.get("unit")}
    return {"filename": label.filename, "fields": values}


@app.post("/register")
async def register(source: UploadFile = File(...), reference: UploadFile = File(...)) -> dict:
    try:
        source_image = decode_image(await source.read())
        reference_image = decode_image(await reference.read())
        result = register_images(source_image, reference_image)
        return {"metrics": result.metrics, "transform": result.transform.tolist(), "aligned_preview": encode_png(result.aligned), "matches_preview": encode_png(result.matches_preview), "source_preview": encode_png(source_image), "reference_preview": encode_png(reference_image), "tie_points": result.tie_points}
    except RegistrationError as exc:
        raise HTTPException(422, str(exc)) from exc
