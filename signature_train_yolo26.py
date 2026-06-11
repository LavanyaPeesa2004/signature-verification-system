from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")  # or yolo11n.pt if yolo26 weights are unavailable

    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project="runs",
        name="sig",
        plots=False  
    )

if __name__ == "__main__":
    main()