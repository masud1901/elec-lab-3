import argparse
from pathlib import Path

DEFAULTS = dict(
    # ---- data ----
    data_dir="data",            # where FER2013 is looked for / downloaded to
    labels="ferplus",           # "ferplus" (8 classes, cleaner labels) or "fer2013" (original 7 classes)
    ferplus_mode="majority",    # "majority" (one label per image) or "soft" (train on the annotators' vote shares)
    # ---- model ----
    model="resnet18",           # "resnet18", "alexnet", "small_cnn" or "vit_face" (pretrained on expressions)
    pretrained=True,            # start from ImageNet weights (ignored for small_cnn)
    img_size=128,               # FER2013 images are 48x48; pretrained networks work better on upscaled images
    small_stem=False,           # resnet18: shrink the stem so 48x48 inputs are not reduced to 2x2
    freeze_encoder=False,       # vit_face: train only the new classifier head (fits a small GPU)
    # ---- training ----
    epochs=25,
    batch_size=128,
    lr=5e-4,
    weight_decay=1e-4,
    class_weighting="sqrt_inverse",  # "inverse" (paper-style), "sqrt_inverse" (gentler) or "none"
    label_smoothing=0.1,
    augment=True,               # random flip / rotation / shift / brightness / erasing
    tta=True,                   # test-time augmentation: also predict on the mirrored face and average
    mixup=0.0,                  # mixup alpha (0 = off); e.g. 0.2
    cutmix=0.0,                 # CutMix alpha (0 = off); e.g. 1.0
    mix_prob=0.5,               # probability that a batch is mixed at all
    ema_decay=0.0,              # exponential moving average of the weights (0 = off); e.g. 0.999
    select_metric="macro_f1",   # validation score used to pick the best checkpoint: "macro_f1" or "accuracy"
    seed=42,
    # ---- output ----
    out_dir="runs",             # checkpoints, plots and results go to out_dir/<run name>/
)

HELP = dict(
    data_dir="folder containing FER2013 (fer2013.csv, the Kaggle zip, or train/ and test/ folders)",
    labels="ferplus: 8 classes, cleaner labels | fer2013: the original 7 classes",
    ferplus_mode="majority: one label per image | soft: train on the annotators' vote shares",
    model="alexnet is the AffectNet paper's network; small_cnn trains from scratch; "
          "vit_face is a ViT pretrained on facial expressions",
    pretrained="start from ImageNet weights",
    img_size="size images are upscaled to (FER2013 images are 48x48)",
    small_stem="resnet18 with a stem for small images (stride-1 3x3 conv, no max-pool); use with --img-size 48",
    freeze_encoder="vit_face only: freeze the pretrained encoder and train the new head",
    epochs="number of passes over the training set",
    batch_size="images per training step",
    lr="peak learning rate",
    weight_decay="AdamW weight decay",
    class_weighting="weighted loss for class imbalance; inverse is the paper's choice",
    label_smoothing="label smoothing for the loss",
    augment="random flip / rotation / shift / brightness / erasing during training",
    tta="also predict on the mirrored face and average",
    mixup="mixup alpha: blend two training images and their labels (0 = off)",
    cutmix="CutMix alpha: paste a patch of one image into another (0 = off)",
    mix_prob="probability that a training batch is mixed with mixup or CutMix",
    ema_decay="keep an exponential moving average of the weights and evaluate that (0 = off)",
    select_metric="validation score used to pick the best checkpoint",
    seed="random seed",
    out_dir="results go to out_dir/<run name>/",
)

CHOICES = dict(
    labels=["ferplus", "fer2013"],
    ferplus_mode=["majority", "soft"],
    model=["resnet18", "alexnet", "small_cnn", "efficientnet_b0", "vit_face"],
    class_weighting=["inverse", "sqrt_inverse", "none"],
    select_metric=["macro_f1", "accuracy"],
)


def get_config(argv=None):
    parser = argparse.ArgumentParser(description="AffectNet-style facial expression baseline on FER2013")
    for name, default in DEFAULTS.items():
        flag = "--" + name.replace("_", "-")
        help_text = f"{HELP.get(name, '')} (default: {default})"
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default, help=HELP.get(name))
        else:
            parser.add_argument(flag, type=type(default), default=default, choices=CHOICES.get(name), help=help_text)
    cfg = parser.parse_args(argv)
    cfg.data_dir = Path(cfg.data_dir)
    cfg.run_dir = Path(cfg.out_dir) / run_name(cfg)
    return cfg


def run_name(cfg):
    """Folder name. The default settings keep the documented name; anything else is appended.

    Appending the non-default settings means two experiments can never overwrite each other's checkpoints
    (train.py resumes automatically from last.pt, which would otherwise silently continue the wrong run).
    """
    mode = f"_{cfg.ferplus_mode}" if cfg.labels == "ferplus" else ""
    name = f"{cfg.model}_{cfg.labels}{mode}_{cfg.img_size}px_{cfg.class_weighting}"
    extras = []
    if cfg.epochs != DEFAULTS["epochs"]:
        extras.append(f"e{cfg.epochs}")
    if cfg.lr != DEFAULTS["lr"]:
        extras.append(f"lr{cfg.lr:g}")
    if cfg.batch_size != DEFAULTS["batch_size"]:
        extras.append(f"bs{cfg.batch_size}")
    if cfg.small_stem:
        extras.append("smallstem")
    if cfg.freeze_encoder:
        extras.append("frozen")
    if cfg.mixup:
        extras.append(f"mixup{cfg.mixup:g}")
    if cfg.cutmix:
        extras.append(f"cutmix{cfg.cutmix:g}")
    if cfg.ema_decay:
        extras.append(f"ema{cfg.ema_decay:g}")
    if cfg.label_smoothing != DEFAULTS["label_smoothing"]:
        extras.append(f"ls{cfg.label_smoothing:g}")
    if not cfg.augment:
        extras.append("noaug")
    if not cfg.pretrained:
        extras.append("scratch")
    if not cfg.tta:
        extras.append("notta")
    if cfg.seed != DEFAULTS["seed"]:
        extras.append(f"s{cfg.seed}")
    return name + ("_" + "_".join(extras) if extras else "")
