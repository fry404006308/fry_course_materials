import cv2
from ultralytics import YOLO

from pathlib import Path


# 模型路径
model_path = Path('../model').resolve()
# 资源路径
asset_path = Path('../assets').resolve()


# Load the YOLOv8 model
model = YOLO(model_path / 'yolov8n.pt')

# Open the video file
video_path = str(asset_path / "01.mp4")
print(f"video_path : {video_path}")
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()