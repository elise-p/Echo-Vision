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
results = model('https://th-thumbnailer.cdn-si-edu.com/aSh561F_GKG2lpzgEmvYnb0Mxtc=/fit-in/1200x0/https://tf-cmsv2-smithsonianmag-media.s3.amazonaws.com/filer/30/15/3015dc19-dd2f-430f-8913-4af9b7a99e2c/abbey_road.jpg')
results[0].show()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model