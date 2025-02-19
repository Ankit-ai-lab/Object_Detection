from ultralytics import YOLO
import torch
import cv2
import numpy as np
from google.colab.patches import cv2_imshow  

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 model
model = YOLO("yolov8n.pt").to(device)

# Load video file
video_path = "/content/WhatsApp Video 2025-02-20 at 00.03.38_b9de1757.mp4"  # Ensure this file is uploaded
cap = cv2.VideoCapture(video_path)

frame_width = 640
frame_height = 480

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Run YOLO on frame
    results = model(frame, device=device)
    vehicle_count = len(results[0].boxes)

    # Draw bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display vehicle count
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show frame (Colab method)
    cv2_imshow(frame)

cap.release()
cv2.destroyAllWindows()
