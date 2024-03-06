""" This example demonstrates validating a model trained model with QAT."""
from ultralytics import YOLO
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained model
model = YOLO('qat_models/w4a4/w4a4.pt').to(device)

# Wrap DetectionModel with LiteML
retrainer_cfg = RetrainerConfig('qat_models/w4a4/liteml_w4a4.yaml')
model.model = RetrainerModel(model.model, config=retrainer_cfg).to(device)

# Validate the model
metrics = model.val(data='coco_ailabs.yaml')
