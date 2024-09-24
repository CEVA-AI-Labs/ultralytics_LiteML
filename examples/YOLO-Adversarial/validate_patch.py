from PIL import Image, ImageDraw
from ultralytics import YOLO
from train_patch import PatchApplier, PatchTransformer
from torchvision import transforms
import os
import numpy as np
import torch
from test_patch import PatchAttack, pad_and_scale
from matplotlib import pyplot as plt
COCO = 'coco_ailabs.yaml'


def save_results(txt_path, results, img_size):
    textfile = open(txt_path, 'w+')
    boxes = results[0].boxes
    results = []
    for box in boxes:
        cls_id = box.cls.item()
        if (cls_id == 0):  # if person
            x_center = box.xywh[0, 0].item() / img_size
            y_center = box.xywh[0, 1].item() / img_size
            width = box.xywh[0, 2].item() / img_size
            height = box.xywh[0, 3].item() / img_size
            textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
            results.append([cls_id, x_center, y_center, width, height])
    textfile.close()
    results = np.array(results)
    return results


def main():


    print("Setting everything up")
    imgdir = "../../inria/Test/pos"
    savedir = "inria_test"

    model = YOLO('yolov8n.pt').cuda()  # load an official model
    attack = PatchAttack(model)

    batch_size = 1
    max_lab = 14
    img_size = 640

    patch_size = 300

    clean_results = []
    noise_results = []
    patch_results = []

    print("Done")
    # Loop over cleane beelden
    for i, imgfile in enumerate(os.listdir(imgdir)):
        print("new image")
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]  # image name w/o extension
            txtname = name + '.txt'
            txtpath = os.path.abspath(os.path.join(savedir, 'clean/', 'yolo-labels/', txtname))

            # open beeld en pas aan naar yolo input size
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')

            if os.path.exists(txtpath):       #check to see if label file contains data.
                label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            img, label = pad_and_scale(img, label, img_size)

            results_clean = model.predict(img, imgsz=img_size, save=False, show=False, conf=0.5)
            label = save_results(txtpath, results_clean, img_size)

            attacked_img = attack(img, img_size, label)
            # plt.imshow(attacked_img)
            results_patched = model.predict(attacked_img, save=False, show=False, conf=0.5)

            properpatchedname = name + "_p.png"
            # p_img_pil.save(os.path.join(savedir, 'proper_patched/', properpatchedname))
            txtname = properpatchedname.replace('.png', '.txt')
            txtpath = os.path.abspath(os.path.join(savedir, 'proper_patched/', 'yolo-labels/', txtname))
            save_results(txtpath, results_patched, img_size)
            if i == 4:
                exit()


if __name__ == '__main__':
    main()