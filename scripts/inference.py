from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("runs/detect/new_training/weights/best.pt")  # Path to the trained model

# Run inference on a video
results = model.predict(
    source="videos/fsd1.mp4",  # Input video path
    save=True,                          # Save annotated video
    show=True,                          # Shows the video while running this file
    conf=0.25,                          # Confidence minimum - minimal confidence needed to show label
)
