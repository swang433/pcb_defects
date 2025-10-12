from ultralytics import YOLO

model = YOLO(r"D:\Code\ML_Python\defect_det\training_runs\pcb_defects_v1_bigger\weights\best.pt")

# Test on all val images
results = model.predict(
    source=r"D:\Code\ML_Python\defect_det\PCB_DATASET\yolo_dataset\val\images",
    save=True,
    conf=0.5, 
    project=r"D:\Code\ML_Python\defect_det\inference_results",
    name="defect_det_v8s"
)

print("Predictions saved to inference_results/yolov8s_results_75e/")