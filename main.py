from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import supervision as sv
from inference import get_model
import os

app = FastAPI()

# Permitir peticiones desde Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar modelo y tracker
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
model = get_model(model_id="contador-de-sombras-de-peces/1", api_key=ROBOFLOW_API_KEY)
tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30)

# Línea de conteo (ajusta X segun el ancho de tu canvas/video)
line_zone = sv.LineZone(start=sv.Point(x=1080, y=0), end=sv.Point(x=1080, y=1116))

@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    line_zone.trigger(detections)

    return {
        "count": line_zone.in_count + line_zone.out_count,
        "tracked_objects": [
            {"id": int(tid), "bbox": bbox.tolist()}
            for bbox, tid in zip(detections.xyxy, detections.tracker_id)
        ]
    }