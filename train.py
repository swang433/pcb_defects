from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')

    training_run_name = input('Enter name of the current training run: ')
    
    results = model.train(
        data="PCB_DATASET/pcb_defects.yaml",
        epochs=75,
        imgsz=640,
        batch=16,
        workers=0,
        project="training_runs",
        name=training_run_name
    )
    
    print("Training complete!")
    print(f"Results saved to: {results.save_dir}")