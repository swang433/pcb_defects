from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI(title="PCB Defect Detection API")

model = YOLO(r"D:\Code\ML_Python\defect_det\training_runs\pcb_defects_v1_bigger\weights\best.pt")

@app.post("/predict")
async def predict_defects(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img, conf=0.5)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            detections.append({
                'class': r.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })
    
    return {
        'num_defects': len(detections),
        'defects': detections
    }

@app.get("/")
def root():
    return {"status": "online", "model": "YOLOv8n PCB Defects"}