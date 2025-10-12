from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')
    
    results = model.train(
        data=r"D:\Code\ML_Python\defect_det\PCB_DATASET\pcb_defects.yaml",
        epochs=75,
        imgsz=640,
        batch=16,
        workers=0,
        project=r"D:\Code\ML_Python\defect_det\training_runs",
        name='pcb_defects_v1_bigger'
    )
    
    print("Training complete!")
    print(f"Results saved to: {results.save_dir}")