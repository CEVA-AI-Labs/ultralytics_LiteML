import sys
import torch
from attacks import FGSM, PGD
from adversarial import v8Losses
import argparse
import os
import csv
sys.path.append('/projects/vbu_projects/users/royj/gitRepos/ailabs_liteml')
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='name of the model [yolov3.pt, yolov5n.pt, yolov8n.pt] or path to QAT model.')
    parser.add_argument('--cfg', type=str, default='configs/w4a4.yaml',
                        help='path to configuration yaml file')
    parser.add_argument('--quantization_type', type=str, default='PTQ',
                        help='Quantization type (PTQ, QAT, float).')
    parser.add_argument('--attack', type=bool, default=False,
                        help='Apply adversarial attack (PGD).')
    parser.add_argument('--results_path', type=str, default='val_default',
                        help='Apply adversarial attack (PGD).')

    args = parser.parse_args()
    return args


# def create_output_file_name(args):
#     model_name = args.model.replace('.pt', '')
#     cfg = args.cfg.split('/')[-1].replace('.yaml', '')
#     attack = args.attack
#     output_path = os.path.join(model_name, cfg)
#     if attack:
#         output_path = f'{output_path}_e-{args.epochs}_f-{str(args.fraction)}_attack'
#     else:
#         output_path = f'{output_path}_e-{args.epochs}_f-{str(args.fraction)}'
#     return output_path


def save_results(metrics, results_path):
    map_dict = [{'mAP50_95': metrics.box.map, 'mAP50': metrics.box.map50},]
    fields = ['mAP50_95', 'mAP50']
    results_path = os.path.join('runs/detect', results_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    filename = os.path.join(results_path, 'map.csv')
    # writing to csv file
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(map_dict)


def main():
    args = get_args()
    quantization_type = args.quantization_type
    apply_attack = args.attack
    model_path = args.model
    cfg_path = args.cfg
    results_path = args.results_path

    # quantization_type = 'PTQ'  # Choose 'QAT' or 'PTQ'
    # apply_attack = False

    model = YOLO(model_path).cuda()  # load an official model for PTQ
    # model = YOLO('yolov8n.pt').cuda()  # load an official model for PTQ
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
        retrainer_cfg = RetrainerConfig(cfg_path)
        # retrainer_cfg = RetrainerConfig("configs/w4a4_per_channel_per_channel.yaml")
        # retrainer_cfg.optimizations_config['QAT']['weights_quantization']['quant_and_freeze'] = True
        model.model = RetrainerModel(model.model, config=retrainer_cfg).cuda()
        # forward pass random image in the model to update scale factor shapes
        inp = torch.rand((1, 3, 640, 640)).cuda()
        out = model.model._model._model(inp)

    if apply_attack:
        # attack = FGSM(model)
        attack = PGD(model)
        metrics = model.val(data='coco_ailabs.yaml', attack=attack, name=results_path)
    else:
        metrics = model.val(data='coco_ailabs.yaml', name=results_path)

    # Save results
    save_results(metrics, results_path)
    print('Done')


if __name__ == '__main__':
    main()