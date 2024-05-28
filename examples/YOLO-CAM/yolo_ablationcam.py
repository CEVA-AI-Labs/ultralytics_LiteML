import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import torchvision
from pytorch_grad_cam import AblationLayer
from ultralytics.utils.ops import non_max_suppression
from ultralytics import YOLO
from matplotlib import pyplot as plt

COLORS = np.random.uniform(0, 255, size=(80, 3))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class YOLOTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        if model_outputs.dim() == 2:
            model_outputs = model_outputs.unsqueeze(0)
        model_outputs = non_max_suppression(
                    model_outputs,
                    conf_thres=0.25,
                    iou_thres=0.7,
                    labels=[],
                    multi_label=True,
                    agnostic=False,
                    max_det=300,
                )
        model_boxes, model_cls, model_labels = parse_detections_yolo(model_outputs)
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()
        elif torch.backends.mps.is_available():
            output = output.to("mps")

        if model_boxes.shape[0] == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = box[None, :]
            if torch.cuda.is_available():
                box = box.cuda()
                model_boxes = model_boxes.cuda()

            ious = torchvision.ops.box_iou(box, model_boxes)
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_labels[index] == label:
                score = ious[0, index] + model_cls[index]
                # score = ious[0, index]
                # score = model_cls[index]
                output = output + score
        return output


class AblationLayerYOLO(AblationLayer):
    def __init__(self, target_layers):
        super(AblationLayer, self).__init__()
        # set f and i according to f and i values of the replaced layer (C3 in this case)
        self.f = target_layers[0].f
        self.i = target_layers[0].i

    def __call__(self, x):
        output = self.activations
        for i in range(output.size(0)):
            # Commonly the minimum activation will be 0,
            # And then it makes sense to zero it out.
            # However depending on the architecture,
            # If the values can be negative, we use very negative values
            # to perform the ablation, deviating from the paper.
            output[i, self.indices[i], :] = -1

        return output


def parse_detections_yolo(results):
    detections = results[0]
    boxes = detections[:, :4].to(int)
    cls = detections[:, 4]
    labels = detections[:, 5]
    return boxes, cls, labels

def draw_detections(boxes, labels, img):
    labels = labels.detach().cpu().numpy().astype(int)
    boxes = boxes.detach().cpu().numpy().astype(int)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            COLORS[label],
            2)

        cv2.putText(img, coco_names[label], (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[label], 2,
                    lineType=cv2.LINE_AA)
    return img


def renormalize_cam_in_bounding_boxes(boxes, labels, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis=0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_detections(boxes, labels, eigencam_image_renormalized)
    return image_with_bounding_boxes


image_path = "dog_cat_3.jpg"
img = np.array(Image.open(image_path))
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()
img = np.float32(img) / 255
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)

model = YOLO('yolov5s.pt')
coco_names = model.names
model.model.eval()

model = model.to(device)
tensor = tensor.to(device)

preds = model.model(tensor)
results = non_max_suppression(
            preds,
            conf_thres=0.25,
            iou_thres=0.7,
            labels=[],
            multi_label=True,
            agnostic=False,
            max_det=300,
        )
boxes, cls, labels = parse_detections_yolo(results)
detections = draw_detections(boxes, labels, rgb_img.copy())
plt.figure(1)
plt.imshow(detections)
# Image.fromarray(detections).show()


targets = [YOLOTarget(labels=labels, bounding_boxes=boxes)]
# target_layers = [model.model.model[-2]]  # f=-1, i=23
# target_layers = [model.model.model[-5]]  # f=-1, i=20
# target_layers = [model.model.model[-8]]  # f=-1, i=17
# target_layers = [model.model.model[-8], model.model.model[-5]]
target_layers = [model.model.model[-8], model.model.model[-5], model.model.model[-2]]

cam = AblationCAM(model.model,
                  target_layers,
                  ablation_layer=AblationLayerYOLO(target_layers=target_layers),
                  ratio_channels_to_ablate=1.0)

grayscale_cam = cam(tensor, targets=targets)[0, :, :]

cam_image = renormalize_cam_in_bounding_boxes(boxes, labels, img, grayscale_cam)  # with renormalization
# cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)  # without renormalization
# cam_image = draw_detections(boxes, labels, cam_image)  # without renormalization

# Image.fromarray(cam_image).show()
plt.figure(2)
plt.imshow(cam_image)
plt.show()

print('Done')
