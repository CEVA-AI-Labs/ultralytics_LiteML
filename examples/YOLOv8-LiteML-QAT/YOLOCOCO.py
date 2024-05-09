import torch
from ultralytics.data import build_dataloader, build_yolo_dataset, YOLODataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataset(model, dataset_yaml='coco_ailabs.yaml', batch_size=16, fraction=0.01):
    dataset, calibration_loader = load_dataset(model, data=dataset_yaml, imgsz=640, batch_size=batch_size, device=device, fraction=fraction)
    def calibration_loader_key(model, x):
        x = x["img"].float().to(device) / 255
        return model(x)
    return calibration_loader, calibration_loader_key


def load_dataset(model, cfg=DEFAULT_CFG,  batch_size=16, **kwargs):
    args = {**model.overrides, **kwargs, "mode": "train"}  # highest priority args on the right
    cfg = get_cfg(cfg, args)
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
    return YOLODataset(
        img_path=img_path,
        imgsz=args.imgsz,
        batch_size=batch,
        augment=False,  # augmentation
        hyp=args,  # TODO: probably add a get_hyps_from_cfg function
        rect=args.rect or (mode == "val"),  # rectangular batches
        cache=args.cache or None,
        single_cls=args.single_cls or False,
        stride=int(gs),
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=args.task,
        classes=args.classes,
        data=data,
        fraction=args.get('fraction', 1.0),
    )


def get_dataloader(dataset, batch_size=16, rank=0, mode="train", workers=0):
    """Construct and return dataloader."""
    assert mode in ["train", "val"]
    shuffle = mode == "train"
    if getattr(dataset, "rect", False) and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    workers = workers if mode == "train" else workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader