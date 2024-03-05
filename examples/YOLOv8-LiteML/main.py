from ultralytics import YOLO
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig, calibaration_split
import sys, os

# # print("script: cwd is", repr(os.getcwd()))
# # Get the path of the current script
# script_path = os.path.abspath(__file__)
# # Get the directory of the script
# script_dir = os.path.dirname(script_path)
# # Change the current working directory to the script's directory
# os.chdir(script_dir)
# # print("script: cwd is", repr(os.getcwd()))


# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('runs/detect/train/weights/best.pt').cuda()  # load an official model
# retrainer_cfg = RetrainerConfig("/projects/vbu_projects/users/royj/LinuxProjects/YOLOv8/YOLOv8/liteml_config.yaml")

# if resume training a model, wrapping with Retrainer Model is not needed.
# retrainer_cfg = RetrainerConfig("liteml_config.yaml")
# # model.eval()
# model.model = RetrainerModel(model.model, config=retrainer_cfg).cuda()

# TODO: we have to modify the DetectionModel (model.model) yaml file here to support liteml quantized modules.
print('Training model')
results = model.train(data='coco_ailabs.yaml', epochs=10, imgsz=640, resume=True)
# results = model.train(data='coco_ailabs.yaml', epochs=10, imgsz=640, save_period=1, fraction=0.01, device=0) # train on multiple gpus device=[0, 1]

# Resume training


# Validate the model
metrics = model.val(data='coco_ailabs.yaml')  # no arguments needed, dataset and settings remembered

print('Done')