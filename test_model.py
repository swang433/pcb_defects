from ultralytics import YOLO
from pathlib import Path

model_name = input('Enter model name: ')
model_path = Path("training_runs") / model_name / "weights" / "best.pt"
model = YOLO(model_path)

# Test on all val images
results = model.predict(
    source=Path("PCB_DATASET/yolo_dataset/val/images"),
    save=True,
    conf=0.5, 
    project="inference_results",
    name="defect_det_v8s"
)

print("Predictions saved to inference_results/defect_det_v8s/")