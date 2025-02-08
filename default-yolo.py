from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=5,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model('https://images.pexels.com/photos/139303/pexels-photo-139303.jpeg?cs=srgb&dl=pexels-joshsorenson-139303.jpg&fm=jpg')
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model