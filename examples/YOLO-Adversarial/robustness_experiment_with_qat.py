from ultralytics import YOLO
import torch
import torch.nn as nn
from attacks import FGSM, PGD
from adversarial import v8Losses
import copy
from matplotlib import pyplot as plt
import argparse
import sys
sys.path.append('/projects/vbu_projects/users/royj/gitRepos/ailabs_liteml')
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig


CONFS_PC_PC = {
    4: 'configs/w4a4_per_channel_per_channel.yaml',
    5: 'configs/w5a5_per_channel_per_channel.yaml',
    6: 'configs/w6a6_per_channel_per_channel.yaml',
    7: 'configs/w7a7_per_channel_per_channel.yaml',
    8: 'configs/w8a8_per_channel_per_channel.yaml',
}
CONFS_W_ONLY = {
    4: 'configs/weight_only/w4_per_channel.yaml',
    5: 'configs/weight_only/w5_per_channel.yaml',
    6: 'configs/weight_only/w6_per_channel.yaml',
    7: 'configs/weight_only/w7_per_channel.yaml',
    8: 'configs/weight_only/w8_per_channel.yaml',
}
CONFS_PC_PT = {
    4: 'configs/w4a4_per_channel_per_tensor.yaml',
    5: 'configs/w5a5_per_channel_per_tensor.yaml',
    6: 'configs/w6a6_per_channel_per_tensor.yaml',
    7: 'configs/w7a7_per_channel_per_tensor.yaml',
    8: 'configs/w8a8_per_channel_per_tensor.yaml',
}
MODELS_QAAT = {
    'float': 'runs/detect/yolov5n/QAAT/float_e-3_f-1.0_AT/weights/best.pt',
    4: 'runs/detect/yolov5n/QAAT/w4a4_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt',
    5: 'runs/detect/yolov5n/QAAT/w5a5_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt',
    6: 'runs/detect/yolov5n/QAAT/w6a6_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt',
    7: 'runs/detect/yolov5n/QAAT/w7a7_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt',
    8: 'runs/detect/yolov5n/QAAT/w8a8_per_channel_per_channel_e-3_f-1.0_AT/weights/best.pt',
}

COCO = 'coco_ailabs.yaml'


def get_args():
    parser = argparse.ArgumentParser(description='Robustness test arguments')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLO model. Choose from: [yolov3.pt, yolov5n.pt, yolov8n.pt]')
    parser.add_argument('--attacked_model', type=str, default='quantized',
                        help="Choose 'quantized' if the adversarial examples are created using the quantized model or 'float' if the examples are created from the float model.")
    parser.add_argument('--quant_granularity', type=str, default='pc_pc',
                        help="Choose 'pc_pc' for weights per-channel and activations per-channel, "
                             "'pc_pt' for for weights per-channel and activations per-tensor or 'w_only' for for weights only quantization.")
    parser.add_argument('--qat', type=bool, default=False,
                        help="If True, evaluate QAT models. If false, evaluate PTQ models.")

    args = parser.parse_args()
    return args


def create_figure_name(yolo_version, attacked_model, quantization_granularity, qat):
    """ create figure name according to the test """
    if 'yolov3' in yolo_version:
        yolo_name = 'yolov3'
    elif 'yolov5n' in yolo_version:
        yolo_name = 'yolov5n'
    elif 'yolov8n' in yolo_version:
        yolo_name = 'yolov8n'
    fig_name = f'{yolo_name}_attacked-model-{attacked_model}_quantization-{quantization_granularity}'
    return fig_name


def get_config(quantization_granularity):
    configs_dict = {'pc_pc': CONFS_PC_PC, 'pc_pt': CONFS_PC_PT, 'w_only': CONFS_W_ONLY, 'qat': MODELS_QAAT}
    return configs_dict.get(quantization_granularity)


def visualize_plots(map_clean, map_float_clean, map_adv, map_float_adv, conf_dict, figure_name):
    x = list(conf_dict.keys())
    plt.figure(1)
    plt.plot(x, map_clean, color='b', marker='o', label='clean')
    plt.axhline(y=map_float_clean, color='b', linestyle='--', label='float clean')
    plt.plot(x, map_adv, color='r', marker='o', label='attacked')
    plt.axhline(y=map_float_adv, color='r', linestyle='--', label='float attacked')
    plt.legend()
    plt.xlabel('bits')
    plt.ylabel('mAP50-95')
    plt.xticks(x)
    plt.savefig(f"runs/robustness_test/{figure_name}.png")
    # plt.show()


def main():
    args = get_args()
    quant_granularity = args.quant_granularity
    yolo_version = args.model
    attacked_model = args.attacked_model
    qat = args.qat

    # quant_granularity = 'pc_pc'  # 'pc_pc' or 'pc_pt' or 'w_only'
    # yolo_version = 'yolov5n.pt'  # yolov3.pt, yolov5n.pt, yolov8n.pt
    # attacked_model = 'quantized'  # Choose 'quantized' if the attack is applied on the quantized model or 'float' if the attack is applied on a float model.

    config_dict = get_config(quant_granularity)
    if qat:
        yolo_version = config_dict.pop('float')

    map_clean = []
    map_adv = []
    fig_name = create_figure_name(yolo_version, attacked_model, quant_granularity, qat)
    # validate float model
    model_float = YOLO(yolo_version).cuda()
    model_float.model.criterion = v8Losses(model_float.model)
    attack = PGD(model_float)
    map_float_clean = (model_float.val(data=COCO, plots=False)).box.map
    map_float_adv = (model_float.val(data=COCO, attack=attack, plots=False)).box.map
    del model_float
    del attack

    if not qat:
        model_original = YOLO(yolo_version).cuda()
    for cfg_file in config_dict.values():
        if qat:
            model_original = YOLO(cfg_file).cuda()
        model = copy.deepcopy(model_original)
        if isinstance(model.model, RetrainerModel):
            # Model is wrapped twice with LiteML RetrainerModel class
            model.model._model._model.criterion = v8Losses(model.model._model._model)
        else:
            model.model.criterion = v8Losses(model.model)

        if attacked_model == 'float':
            attack = PGD(copy.deepcopy(model))
        if not qat:
            retrainer_cfg = RetrainerConfig(cfg_file)
            model.model = RetrainerModel(model.model, config=retrainer_cfg).cuda()
            inp = torch.rand((1, 3, 640, 640)).cuda()
            # forward pass random image in the model to update scale factor shapes
            out = model.model._model._model(inp)
        model_adv = copy.deepcopy(model)

        if attacked_model == 'quantized':
            # Apply attacks on the quantized model directly
            attack = PGD(model_adv)

        # Validate the model
        metrics_clean = model.val(data=COCO, plots=False)
        metrics_attacked = model_adv.val(data=COCO, attack=attack, plots=False)
        map_clean.append(metrics_clean.box.map)  # map50-95
        map_adv.append(metrics_attacked.box.map)  # map50-95

        del model
        del model_adv
        del attack

    print(map_clean)
    print(map_adv)
    visualize_plots(map_clean, map_float_clean, map_adv, map_float_adv, config_dict, figure_name=fig_name)


if __name__ == '__main__':
    main()