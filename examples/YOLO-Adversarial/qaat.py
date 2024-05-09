from ultralytics import YOLO
import torch
import argparse
import os
import sys
sys.path.append('/projects/vbu_projects/users/royj/gitRepos/ailabs_liteml')
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
from attacks import FGSM, PGD
from adversarial import v8Losses

def get_args():
    parser = argparse.ArgumentParser(description='QAT arguments')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='name of the model [yolov3.pt, yolov5n.pt, yolov8n.pt]')
    parser.add_argument('--cfg', type=str, default='configs/w4a4.yaml',
                        help='path to configuration yaml file')
    # parser.add_argument('--results_path', type=str, default='yolov5n/w4a4',
    #                     help='path for saving the qat output')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs for retraining')
    parser.add_argument('--fraction', type=float, default=1,
                        help='fraction of the training dataset to use for retraining')
    parser.add_argument('--at', type=bool, default=False,
                        help='Apply adversarial training if True.')

    args = parser.parse_args()
    return args


def create_output_file_name(args):
    model_name = args.model.replace('.pt', '')
    cfg = args.cfg.split('/')[-1].replace('.yaml', '')
    adversarial_training = args.at
    output_path = os.path.join(model_name, cfg)
    if adversarial_training:
        output_path = f'{output_path}_e-{args.epochs}_f-{str(args.fraction)}_AT'
    else:
        output_path = f'{output_path}_e-{args.epochs}_f-{str(args.fraction)}'
    return output_path


def main():
    args = get_args()
    batch_size = 16
    results_path = create_output_file_name(args)
    # Load a model
    model = YOLO(args.model).cuda()
    # For adversarial attacks
    # model.model.criterion = v8Losses(model.model)
    if args.at:
        attack = PGD(model)
    else:
        attack = None

    if args.cfg != 'float':
        retrainer_cfg = RetrainerConfig(args.cfg)
        print(f'************** Retraining {args.cfg} **************')
        # model.eval()
        model.model = RetrainerModel(model.model, config=retrainer_cfg).cuda()

        # forward pass random image in the model to update scale factor shapes
        inp = torch.rand((batch_size, 3, 640, 640)).cuda()
        out = model.model._model._model(inp)

    # Train the model
    results = model.train(data='coco_ailabs.yaml',
                          epochs=args.epochs,
                          batch=batch_size,
                          imgsz=640,
                          save_period=1,
                          fraction=args.fraction,
                          name=results_path,
                          device=0, attack=attack)


if __name__=='__main__':
    main()
