from ultralytics import YOLO

# Load a model
model = YOLO('./230913_1643_train5/weights/best.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model.predict(r'./imgs/butterfly_15.jpg',show=True,save=True)  # predict on an image


print(type(results))
print(len(results))
print(type(results[0]))

boxes = results[0].boxes
masks = results[0].masks
probs = results[0].probs
print(f"boxes:{boxes}")
print(f"masks:{masks }")
print(f"probs:{probs}")


