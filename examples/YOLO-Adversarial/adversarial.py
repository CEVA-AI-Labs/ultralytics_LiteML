from ultralytics import YOLO
import sys, os
import torch
import torch.nn as nn
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.utils.loss import BboxLoss
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from attacks import FGSM, PGD


class v8Losses:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        # self.hyp['box'] = 7.5  # box gain 7.5
        # self.hyp['cls'] = 0.5  # cls gain 0.5
        # self.hyp['dfl'] = 1.5  # dfl gain 1.5
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        # loss[0] *= self.hyp['box']  # box gain 7.5
        # loss[1] *= self.hyp['cls']  # cls gain 0.5
        # loss[2] *= self.hyp['dfl']  # dfl gain 1.5

        return loss[1], loss.detach()  # loss(box, cls, dfl)
        # return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


def load_dataset(model, cfg=DEFAULT_CFG, batch_size=16, **kwargs):
    args = {**model.overrides, **kwargs, "mode": "train"}  # highest priority args on the right
    # overrides = model.overrides
    cfg = get_cfg(cfg, args)
    # overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
    data = check_det_dataset(cfg.data)
    if "yaml_file" in data:
        cfg.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
    trainset = data.get("train")
    dataset = build_dataset(model=model.model, args=cfg, data=data, img_path=trainset, mode="val", batch=batch_size)
    dataloader = get_dataloader(dataset, batch_size, rank=RANK, workers=cfg.workers)
    return dataset, dataloader

def build_dataset(model, args, data, img_path, mode="train", batch=None):
    """
    Build YOLO Dataset.

    Args:
        img_path (str): Path to the folder containing images.
        mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
        batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
    """
    gs = max(int(de_parallel(model).stride.max() if model else 0), 32)
    return build_yolo_dataset(args, img_path, batch, data, mode=mode, rect=mode == "val", stride=gs)


def get_dataloader(dataset, batch_size=16, rank=0, mode="train", workers=0):
    """Construct and return dataloader."""
    assert mode in ["train", "val"]
    shuffle = mode == "train"
    if getattr(dataset, "rect", False) and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    workers = workers if mode == "train" else workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader


def main():
    model = YOLO('yolov8n.pt').cuda()  # load an official model
    # set criterion for calculating the desired loss function for the adversarial attack
    model.model.criterion = v8Losses(model.model)
    # load images
    dataset, dataloader = load_dataset(model, epochs=1, data='coco_ailabs.yaml', imgsz=640, save_period=1, fraction=0.01, device=0)

    inputs = next(iter(dataloader))

    # Convert BGR to RGB
    inputs["img"] = inputs["img"][:, [2, 1, 0]]
    # Convert to float between 0 and 1
    inputs["img"] = inputs["img"].cuda().float() / 255
    inputs["img"].requires_grad = True

    #FGSM
    # fgsm = FGSM(model)
    # perturbed_image = fgsm(inputs)

    # PGD
    pgd = PGD(model)
    perturbed_image = pgd(inputs)

    model.predict(perturbed_image, save=True, show=True, conf=0.5)

    print('Done')

if __name__ == '__main__':
    main()