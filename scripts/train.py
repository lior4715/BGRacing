from ultralytics import YOLO
import os

pretrain_weights = "runs/detect/new_training/weights/best.pt"
# Load the YOLO model (pre-trained weights)
if os.path.exists(pretrain_weights):
    model = YOLO("runs/detect/new_training/weights/best.pt")

else:
    model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano
# Train the model
model.train(
    data="datasets/data.yaml",  # Path to data.yaml
    epochs=5,                  # Number of epochs
    imgsz=640,                  # Image size
    name="new_training",       # Name of the training run
    resume=True,
)
