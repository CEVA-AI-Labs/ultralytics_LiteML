from ultralytics import YOLO
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a model
model = YOLO('yolov3.pt').to(device)

# Wrap DetectionModel with LiteML
retrainer_cfg = RetrainerConfig('liteml_config.yaml')
inp = torch.rand((1, 3, 640, 640)).to(device)
model.model = RetrainerModel.from_pretrained(model.model,
                                             liteml_config_yaml='liteml_config.yaml',
                                             state_dict='yolov3_state_dict.pt',
                                             device=device, dummy_input=inp)

# Validate the model
metrics = model.val(data='coco_ailabs.yaml')

print('Done')