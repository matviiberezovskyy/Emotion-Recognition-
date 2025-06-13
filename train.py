from ultralytics import YOLO
import os
from IPython import display

display.clear_output()

from roboflow import Roboflow

if os.path.exists('data.yaml') is False:
    rf = Roboflow(api_key="NgJiIfMDv7ZDGOgq9wtR")
    project = rf.workspace("lyanhvini").project("emotion-detection-a5i5h")
    version = project.version(4)
    dataset = version.download("yolov8")

train_mode = 0
epochs = 10
base_model_path = "./models/em_det_yolo_v1.pt"
if os.path.exists(base_model_path):
    model = YOLO(base_model_path)  # load a custom pretrained model
else:
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="data.yaml", epochs=epochs, imgsz=640, device='cpu')

