from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo12n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("YOLO/data/13159549_3840_2160_30fps.mp4",save = True,show=True)