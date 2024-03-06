""" This example demonstrates how to perform QAT on a pretrained model."""
from ultralytics import YOLO
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Wrap DetectionModel with LiteML
retrainer_cfg = RetrainerConfig("liteml_config.yaml")
model.model = RetrainerModel(model.model, config=retrainer_cfg).to(device)

# Train the model
results = model.train(data='coco_ailabs.yaml', epochs=10, imgsz=640, save_period=1, fraction=0.01, device=device)

# Validate the model
metrics = model.val(data='coco_ailabs.yaml')
