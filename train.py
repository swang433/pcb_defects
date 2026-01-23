import time
from ultralytics import YOLO

def timer(func): 
    def wrapper(*args, **kwargs): 
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return res
    return wrapper

@timer
def train_model(model, training_run_name, num_epochs):
    results = model.train(
        data="PCB_DATASET/pcb_defects.yaml",
        epochs=num_epochs,
        imgsz=640,
        batch=16,
        workers=0,
        project="training_runs",
        name=training_run_name
    )
    return results

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')
    training_run_name = input('Enter name of the current training run: ')
    num_epochs = int(input('Enter the number of epochs: '))
    results = train_model(model, training_run_name, num_epochs)
    
    print("Training complete!")
    print(f"Results saved to: {results.save_dir}")