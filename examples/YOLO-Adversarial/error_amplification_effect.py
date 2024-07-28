import sys
import torch
from attacks import FGSM, PGD
from adversarial import v8Losses
import argparse
import os
import csv
import copy
from torch import nn, Tensor
from adversarial import load_dataset
from typing import Dict, Iterable, Callable
import functools
sys.path.append('/projects/vbu_projects/users/royj/gitRepos/ailabs_liteml')
from liteml.ailabs_liteml.retrainer import RetrainerModel, RetrainerConfig
from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
torch.manual_seed(0)

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            if isinstance(output, tuple):
                output = output[0][:,4:]
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features


def calc_distances_attack_strength_sweep(model, inputs, epsilons, layer, model_float=None):
    clean_images = inputs["img"]
    model_clean = FeatureExtractor(copy.deepcopy(model), layers=[layer])
    model_adv = FeatureExtractor(copy.deepcopy(model), layers=[layer])
    features_clean = model_clean(clean_images)[layer]
    features_adv_dict = {}
    for eps in epsilons:
        # FGSM
        # fgsm = FGSM(model, eps=eps)
        # perturbed_images = fgsm(inputs)

        # perturbed_images = torch.clip(inputs["img"] + eps*torch.rand_like(inputs["img"]), 0, 1)  # Random noise

        if model_float is None:
            # PGD
            pgd = PGD(model, eps=eps, random_start=True)
            perturbed_images = pgd(copy.deepcopy(inputs))
        else:
            pgd = PGD(model_float, eps=eps, random_start=True)
            perturbed_images = pgd(copy.deepcopy(inputs))

        features_adv_dict[eps] = model_adv(perturbed_images)[layer]
    distances = []
    for eps, features_adv in features_adv_dict.items():
        distance_matrix = torch.linalg.matrix_norm(features_clean - features_adv) / torch.linalg.matrix_norm(
            features_clean)
        # distance_mean = distance_matrix.mean(dim=0)  # average over batch. Each element corresponds to a specific channel
        distance_mean = float(distance_matrix.mean())  # average over batch and channels
        distances.append(distance_mean)
    return distances


def calc_distances_layers_sweep(model, inputs, layers, eps=1, perturbed_images=None):
    clean_images = inputs["img"]
    model_clean = FeatureExtractor(copy.deepcopy(model), layers=layers)
    model_adv = FeatureExtractor(copy.deepcopy(model), layers=layers)
    features_clean = model_clean(clean_images)  # dictionary {layer_name: Tensor, ...}

    if perturbed_images is None:
        # FGSM
        # fgsm = FGSM(model, eps=eps)
        # perturbed_images = fgsm(inputs)

        # PGD
        pgd = PGD(model, eps=eps, random_start=True)
        perturbed_images = pgd(copy.deepcopy(inputs))

    features_adv = model_adv(perturbed_images)  # dictionary {layer_name: Tensor}
    distances = []
    for layer in layers:
        distance_matrix = torch.linalg.matrix_norm(features_clean[layer] - features_adv[layer]) / torch.linalg.matrix_norm(
            features_clean[layer])
        distance_mean = float(distance_matrix.mean())  # average over batch and channels
        distances.append(distance_mean)
    return distances


def layers_test(model, inputs):
    layers_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    layers = [f'model.model.{i}' for i in layers_indices]
    layers_quantized_model = [f'model._model._model.model.{i}' for i in layers_indices]

    # pgd = PGD(model, eps=1, random_start=True)  # Same PGD attack for all models
    # perturbed_images = pgd(copy.deepcopy(inputs))
    # perturbed_images = torch.clip(inputs["img"] + 0.1*torch.rand_like(inputs["img"]), 0, 1)  # Random noise
    perturbed_images = None  # Generate PGD attack for each model separately

    retrainer_cfgs = {
        4: RetrainerConfig('configs/w4a4_per_channel_per_channel.yaml'),
        5: RetrainerConfig('configs/w5a5_per_channel_per_channel.yaml'),
        6: RetrainerConfig('configs/w6a6_per_channel_per_channel.yaml'),
        7: RetrainerConfig('configs/w7a7_per_channel_per_channel.yaml'),
        8: RetrainerConfig('configs/w8a8_per_channel_per_channel.yaml'),
    }

    # retrainer_cfgs = {
    #     4: RetrainerConfig('configs/activation_only/a4_per_channel.yaml'),
    #     8: RetrainerConfig('configs/activation_only/a8_per_channel.yaml'),
    # }
    relative_distances = np.zeros((len(retrainer_cfgs)+1, len(layers)))
    relative_distances_float = calc_distances_layers_sweep(model, inputs, layers, eps=1, perturbed_images=perturbed_images)
    relative_distances[-1] = np.array(relative_distances_float)
    for i, (bits, retrainer_cfg) in enumerate(retrainer_cfgs.items()):
        model_quantized = copy.deepcopy(model)
        model_quantized.model = RetrainerModel(model_quantized.model, config=retrainer_cfg).cuda()
        relative_distances_quantized = calc_distances_layers_sweep(model_quantized, inputs, layers_quantized_model, eps=1, perturbed_images=perturbed_images)
        relative_distances[i] = np.array(relative_distances_quantized)
        del model_quantized

    plt.figure()
    plt.plot(layers_indices, relative_distances.T, marker='o')
    plt.legend(['4 bits', '5 bits', '6 bits', '7 bits', '8 bits', 'Float'])
    # plt.legend(['4 bits', '8 bits', 'float'])

    plt.xlabel('Layer Index')
    plt.ylabel('Normalized Distance')
    plt.show()


def attack_strength_test(model, inputs):
    layer = 'model.model.0'
    layer_quantized_model = 'model._model._model.model.0'

    retrainer_cfgs = {
        4: RetrainerConfig('configs/w4a4_per_channel_per_channel.yaml'),
        5: RetrainerConfig('configs/w5a5_per_channel_per_channel.yaml'),
        6: RetrainerConfig('configs/w6a6_per_channel_per_channel.yaml'),
        7: RetrainerConfig('configs/w7a7_per_channel_per_channel.yaml'),
        8: RetrainerConfig('configs/w8a8_per_channel_per_channel.yaml'),
    }

    # retrainer_cfgs = {
    #     8: RetrainerConfig('configs/activation_only/a8_per_channel.yaml'),
    # }

    epsilons = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8]  # for PGD
    # epsilons = [i * 0.4 for i in range(1, 5)]  # for random

    relative_distances = np.zeros((len(retrainer_cfgs)+1, len(epsilons)))
    # relative_distances_float = calc_distances_attack_strength_sweep(model, inputs, epsilons, layer, model_float=model)
    relative_distances_float = calc_distances_attack_strength_sweep(model, inputs, epsilons, layer)
    relative_distances[-1] = np.array(relative_distances_float)
    for i, (bits, retrainer_cfg) in enumerate(retrainer_cfgs.items()):
        model_quantized = copy.deepcopy(model)
        model_quantized.model = RetrainerModel(model_quantized.model, config=retrainer_cfg).cuda()
        # relative_distances_quantized = calc_distances_attack_strength_sweep(model_quantized, inputs, epsilons, layer_quantized_model, model_float=model)
        relative_distances_quantized = calc_distances_attack_strength_sweep(model_quantized, inputs, epsilons, layer_quantized_model)
        relative_distances[i] = np.array(relative_distances_quantized)
        del model_quantized

    plt.figure()
    plt.plot(epsilons, relative_distances.T, marker='o')
    plt.legend(['4 bits', '5 bits', '6 bits', '7 bits', '8 bits', 'float'])
    # plt.legend(['8 bits', 'float'])
    plt.xlabel(r'Perturbation Strength $\epsilon$')
    plt.ylabel('Normalized Distance')
    plt.show()


if __name__ == '__main__':
    model = YOLO('yolov8n.pt').cuda()  # load an official model for PTQ
    model.model.criterion = v8Losses(model.model)

    # load images
    dataset, dataloader = load_dataset(model, epochs=1, data='coco_ailabs.yaml', imgsz=640, save_period=1, batch_size=128,
                                       fraction=0.01, device=0)
    inputs = next(iter(dataloader))
    # Convert BGR to RGB
    inputs["img"] = inputs["img"][:, [2, 1, 0]]
    # Convert to float between 0 and 1
    inputs["img"] = inputs["img"].cuda().float() / 255
    inputs["img"].requires_grad = True

    # layers_test(model, inputs)
    attack_strength_test(model, inputs)

    print('Done')