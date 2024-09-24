from PIL import Image, ImageDraw
from ultralytics import YOLO
from train_patch import PatchApplier, PatchTransformer
from torchvision import transforms
import os
import numpy as np
import torch


def pad_and_scale(img, lab, out_img_size):
    """

    Args:
        img:

    Returns:

    """
    w, h = img.size
    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(img, (int(padding), 0))
            lab[:, [1]] = (lab[:, [1]] * w + padding) / h
            lab[:, [3]] = (lab[:, [3]] * w / h)
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(img, (0, int(padding)))
            lab[:, [2]] = (lab[:, [2]] * h + padding) / w
            lab[:, [4]] = (lab[:, [4]] * h / w)
    resize = transforms.Resize((out_img_size, out_img_size))
    padded_img = resize(padded_img)  # choose here
    return padded_img, lab


class PatchAttack():
    def __init__(self, model):
        self.model = model
        patchfile = 'saved_patches/patch1.jpg'
        patch_size = 200
        patch_img = Image.open(patchfile).convert('RGB')
        tf = transforms.Resize((patch_size, patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        self.adv_patch = adv_patch_cpu.cuda()
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()

    def __call__(self, img, img_size, label, *args, **kwargs):
        # padded_img, label = pad_and_scale(img, label, img_size)
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        transform = transforms.ToTensor()
        padded_img = transform(img).cuda()
        img_fake_batch = padded_img.unsqueeze(0)
        lab_fake_batch = label.unsqueeze(0).cuda()

        adv_batch_t = self.patch_transformer(self.adv_patch, lab_fake_batch, img_size, do_rotate=True,
                                        rand_loc=False)  # adversarial patch
        # adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False) # random patch
        p_img_batch = self.patch_applier(img_fake_batch, adv_batch_t)
        p_img = p_img_batch.squeeze(0)
        p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
        return p_img_pil


def main_organized():
    img_dir = "../../inria/Train/pos"
    lab_dir = "../../inria/Train/pos/yolo-labels"
    img_name = 'crop001002.png'
    img_size = 640

    img_path = os.path.join(img_dir, img_name)
    lab_path = os.path.join(lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
    model = YOLO('yolov8n.pt').cuda()  # load an official model
    attack = PatchAttack(model)
    img = Image.open(img_path).convert('RGB')
    label = np.loadtxt(lab_path)

    img, label = pad_and_scale(img, label, img_size)
    p_img_pil = attack(img, img_size, label)
    model.predict(p_img_pil, save=True, show=True, conf=0.5)


def main():
    patchfile = 'saved_patches/patch1.jpg'
    # patchfile = 'saved_patches/patch11.jpg'
    # patchfile = 'saved_patches/patch_low_res.jpg'
    img_dir = "../../inria/Train/pos"
    lab_dir = "../../inria/Train/pos/yolo-labels"
    img_name = 'crop001002.png'
    img_size = 640
    patch_size = 200

    img_path = os.path.join(img_dir, img_name)
    lab_path = os.path.join(lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')

    model = YOLO('yolov8n.pt').cuda()  # load an official model

    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer().cuda()

    patch_img = Image.open(patchfile).convert('RGB')

    tf = transforms.Resize((patch_size, patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()
    random_patch = torch.rand(adv_patch_cpu.size()).cuda()

    img = Image.open(img_path).convert('RGB')

    w, h = img.size
    if w == h:
        padded_img = img
    else:
        dim_to_pad = 1 if w < h else 2
        if dim_to_pad == 1:
            padding = (h - w) / 2
            padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
            padded_img.paste(img, (int(padding), 0))
        else:
            padding = (w - h) / 2
            padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
            padded_img.paste(img, (0, int(padding)))
    resize = transforms.Resize((img_size, img_size))
    padded_img = resize(padded_img)


    label = np.loadtxt(lab_path)
    label = torch.from_numpy(label).float()
    if label.dim() == 1:
        label = label.unsqueeze(0)

    transform = transforms.ToTensor()
    padded_img = transform(padded_img).cuda()
    img_fake_batch = padded_img.unsqueeze(0)
    lab_fake_batch = label.unsqueeze(0).cuda()

    adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)  # adversarial patch
    # adv_batch_t = patch_transformer(random_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False) # random patch
    p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
    p_img = p_img_batch.squeeze(0)
    p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())

    model.predict(p_img_pil, save=True, show=True, conf=0.5)
    # model.predict(img, save=True, show=True, conf=0.5)  # original image


if __name__ == '__main__':
    main_organized()
