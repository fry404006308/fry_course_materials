from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('./train3/weights/best.pt')  # load a custom model

# Predict with the model
results = model('../ultralytics/assets/bus.jpg',save=True,project="./")  # predict on an image


"""

Speed: 2.0ms preprocess, 101.1ms inference, 4.6ms postprocess per image at shape (1, 3, 640, 480)
Results saved to predict

进程已结束,退出代码0

"""
