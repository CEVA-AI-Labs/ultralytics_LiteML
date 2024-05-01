from ultralytics import YOLO
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
import torch
import torch.nn as nn
from attacks import FGSM, FGSM2, PGD
from adversarial import v8Losses
import copy


def main():
    quantization_type = 'PTQ'  # Choose 'QAT' or 'PTQ'
    apply_attack = False
    model = YOLO('yolov8n.pt').cuda()  # load an official model for PTQ
    # model = YOLO('runs/detect/yolov5n/QAAT/w4a4_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt').cuda()  # load a QAT model

    # for QAT model
    if quantization_type == 'QAT':
        model.model._model._model.criterion = v8Losses(model.model._model._model)
    # for PTQ model
    else:
        model.model.criterion = v8Losses(model.model)

    # model_float = copy.deepcopy(model)

    # for PTQ model only
    if quantization_type == 'PTQ':
        retrainer_cfg = RetrainerConfig("configs/w4a4_per_channel_per_channel.yaml")
        model.model = RetrainerModel(model.model, config=retrainer_cfg).cuda()
        # forward pass random image in the model to update scale factor shapes
        inp = torch.rand((1, 3, 640, 640)).cuda()
        out = model.model._model._model(inp)

    if apply_attack:
        # attack = FGSM(model)
        attack = PGD(model)
        metrics = model.val(data='coco_ailabs.yaml', attack=attack)
    else:
        metrics = model.val(data='coco_ailabs.yaml')

    print('Done')


if __name__ == '__main__':
    main()