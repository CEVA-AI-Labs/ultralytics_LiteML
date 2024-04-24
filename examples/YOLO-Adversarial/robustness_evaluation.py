from ultralytics import YOLO
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
import torch
import torch.nn as nn
from attacks import FGSM, FGSM2, PGD
from adversarial import v8Losses
import copy


def main():
    model = YOLO('yolov8n.pt').cuda()  # load an official model

    # model.model.criterion = v8Losses(model.model)
    # model_float = copy.deepcopy(model)

    retrainer_cfg = RetrainerConfig("configs/w4a4_per_channel_per_channel.yaml")
    model.model = RetrainerModel(model.model, config=retrainer_cfg).cuda()
    # forward pass random image in the model to update scale factor shapes
    inp = torch.rand((1, 3, 640, 640)).cuda()
    out = model.model._model._model(inp)

    # attack = FGSM(model)
    # attack = PGD(model)


    # Validate the model
    metrics = model.val(data='coco_ailabs.yaml')
    # metrics = model.val(data='coco_ailabs.yaml', attack=attack)

    print('Done')

if __name__ == '__main__':
    main()