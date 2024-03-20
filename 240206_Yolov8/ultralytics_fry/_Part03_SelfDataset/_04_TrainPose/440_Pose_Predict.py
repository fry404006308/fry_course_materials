from ultralytics import YOLO


if __name__ == "__main__":

    # Load a model
    # model = YOLO('yolov8n.pt')  # load an official model
    model = YOLO('./train11/weights/best.pt')  # load a custom model

    # Predict with the model
    results = model('../ultralytics/assets/bus.jpg',save=True)  # predict on an image


"""

Speed: 3.0ms preprocess, 106.0ms inference, 6.0ms postprocess per image at shape (1, 3, 640, 480)
Results saved to D:\_230711_learnArchive\LA_ai_main_CV_classify_detect_segment\runs\pose\predict


"""
