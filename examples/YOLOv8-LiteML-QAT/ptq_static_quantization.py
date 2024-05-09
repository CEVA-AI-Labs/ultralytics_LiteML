from ultralytics import YOLO
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig, calibaration_split
import torch
from YOLOCOCO import get_dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a model
model = YOLO('yolov3.pt').to(device)   # Select from ['yolov3', 'yolov5n', 'yolov8n']

calib_loader, calib_loader_key = get_dataset(model, dataset_yaml='/AI_Labs/datasets/coco/coco.yaml', batch_size=16, fraction=0.01)
#
# Wrap DetectionModel with LiteML
retrainer_cfg = RetrainerConfig('liteml_config.yaml')
retrainer_cfg.optimizations_config["QAT"]["data_quantization"]["calibration_loader"] = calib_loader
retrainer_cfg.optimizations_config["QAT"]["data_quantization"]["calibration_loader_key"] = calib_loader_key
model.model = RetrainerModel(model.model, config=retrainer_cfg).to(device)

# Save state dict
torch.save(model.model.state_dict(), 'yolov3_state_dict.pt')

# forward pass random image in the model to update scale factor shapes
inp = torch.rand((1, 3, 640, 640)).to(device)
out = model.model._model._model(inp)

# Validate the model
metrics = model.val(data='coco_ailabs.yaml')  # no arguments needed, dataset and settings remembered

print('Done')