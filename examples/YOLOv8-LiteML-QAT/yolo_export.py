""" This example demonstrates validating a model trained model with QAT."""
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv as ConvUltra
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
import torch
from YOLOCOCO import get_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained model
model = YOLO('yolov3.pt').to(device)

calib_loader, calib_loader_key = get_dataset(model, dataset_yaml='/AI_Labs/datasets/coco/coco.yaml', batch_size=16, fraction=0.001)
#
# Wrap DetectionModel with LiteML
retrainer_cfg = RetrainerConfig('liteml_config.yaml')
retrainer_cfg.optimizations_config["QAT"]["data_quantization"]["calibration_loader"] = calib_loader
retrainer_cfg.optimizations_config["QAT"]["data_quantization"]["calibration_loader_key"] = calib_loader_key
model.model = RetrainerModel(model.model, config=retrainer_cfg).to(device)

# forward pass random image in the model to update scale factor shapes
inp = torch.rand((1, 3, 640, 640)).cuda()

# Export using LiteML
model.model.export_to_onnx(inp, "yolov3_liteml_old_export_format.onnx", inplace=True)

# Export original float model to ONNX format
# model.export(format='onnx')  # creates 'yolov8n.onnx'