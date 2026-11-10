# Project status — FER2013 / AffectNet baseline exercises

_Last updated: 2026-11-10_

## Goal

Answer the five workshop exercises in `docs/insturtion.md` with a **comparative** table:
macro-F1 is the main score, accuracy is for comparison with published papers. The deadline is tight
(~30 min of GPU time), so all arms run under one equal, reduced budget.

## What has been built

| File | Change | Why |
|---|---|---|
| `config.py` | new options `--small-stem`, `--mixup`, `--cutmix`, `--mix-prob`, `--ema-decay`, `--freeze-encoder`; `run_name()` now appends non-default settings | experiments can no longer overwrite each other's checkpoints, and `train.py` can't silently resume the wrong run |
| `models.py` | `small_stem` for ResNet-18 (3×3/stride-1 stem instead of 7×7/stride-2), new `efficientnet_b0`, new `vit_face` (ViT pretrained on expressions, `trpakov/vit-face-expression`) | exercise 2 (small images), exercise 3 bonus, exercise 4d |
| `augment.py` | `mix_batch()` — mixup and CutMix on images + probability targets | exercise 3 |
| `utils.py` | `ModelEMA`, `using_weights()` (validate with the averaged weights), `cudnn.benchmark` | exercise 3 |
| `train.py` | mixup/CutMix in the loop, EMA updates + EMA validation, EMA weights saved in `best.pt`, **live per-batch progress line**, arm-aware checkpoints | visibility + exercise 3 |
| `evaluate.py` | rebuilds the model with `small_stem`, saves per-class precision/recall/F1 and `classification_report.txt` | needed for 48 px checkpoints and the per-class discussion |
| `predict.py` | rebuilds the model with `small_stem` | exercise 5 |
| `experiments.py` | **new** — the whole experiment matrix in one command: runs every arm, skips finished ones, writes `results_summary.csv`, `per_class_f1.csv`, `RESULTS.md`, `learning_curves_all.png`, `confusion_matrices_all.png` | keeps the Colab notebook unchanged: edit here, push to GitHub, re-run |
| `fer_experiments.ipynb` | (in progress) thin runner: clone/pull repo → run `experiments.py` → show results | Colab execution |

## The experiment matrix (`experiments.py`)

Sprint profile = 8 epochs per arm (16 for the longer-schedule arm), equal splits, equal augmentation.

| Arm | Exercise | Config | Est. on T4 |
|---|---|---|---|
| `A1_ex1_baseline_resnet18_128px` | 1 | ResNet-18, ImageNet, 128 px | ~6 min |
| `A2_ex2_small_stem_48px` | 2 | same, `--img-size 48 --small-stem` | ~3 min |
| `A3_ex3_mixup_cutmix_ema` | 3 | 48 px + mixup 0.2 + CutMix 1.0 + EMA 0.999 + lr 1e-3, 16 epochs | ~6 min |
| `A4_bonus_efficientnet_b0` | 3 | EfficientNet-B0 + mixup + EMA | ~5 min |
| `B1_ex4a_no_pretrain` | 4a | 48 px, `--no-pretrained` | ~3 min |
| `B2_ex4b_alexnet_128px` | 4b | AlexNet, 128 px | ~8 min |
| `B3_ex4d_vit_face` | 4d | ViT expression-pretrained, 112 px, frozen encoder | ~6 min |

## Experiments actually run so far

| Run | Status | Result |
|---|---|---|
| ResNet-18, 128 px, default 25 epochs | **stopped after epoch 5** (GPU freed for Colab work) | best val acc **0.803** / macro-F1 **0.676** at epoch 4; epoch 5 val acc 0.812 / F1 0.644. No test evaluation. Log: `/tmp/opencode/baseline_resnet18.log`, checkpoints in `runs/resnet18_ferplus_majority_128px_sqrt_inverse/` |
| A2 (48 px small stem), 1-epoch smoke test | finished (not a real result) | test acc 0.365 / F1 0.113 after a single epoch — pipeline works end to end |
| A2 repeat, 1 epoch | finished | test acc 0.366 / F1 0.115 — same behaviour |

**No complete exercise result exists yet.** Everything above is either partial or a smoke test.

## Blocking bug (found, not yet fixed)

During training, the validation metrics are wrong:

```
epoch  1/1 | train loss 2.213 | val loss nan | val acc 0.388 | val macro-F1 0.120
TEST accuracy: 0.3661
```

but evaluating the **same saved checkpoint in a fresh process** gives

```
val:  loss 1.843  acc 0.619
test: loss 1.853  acc 0.613
```

So the weights are fine and the in-run evaluation is broken (NaN loss + ~0.39 instead of ~0.62).
This matters a lot: the learning curves, the "best epoch" selection and exercise 4c all depend on
validation. Most likely culprit: `torch.backends.cudnn.benchmark = True` combined with fp16 autocast
(the flag was added for speed and has not yet been ruled out).

## What is left

1. Fix the in-run validation bug and re-verify with a 1-epoch run.
2. Finish `fer_experiments.ipynb` (thin runner that clones/pulls the GitHub repo).
3. Push the repo to GitHub (needed by the notebook).
4. Run the matrix on Colab: `python experiments.py --profile sprint --arms all` (~30 min), or split
   `--arms A` / `--arms B` between the two of us (~15 min each).
5. Paste `RESULTS.md` + the two PNGs into the report; answer exercise 4c from the curves and
   exercise 5 with `predict.py` on our own photos.

## Useful commands

```bash
python experiments.py --dry-run                  # print every command
python experiments.py --arms A --profile sprint  # half the matrix
python experiments.py --skip-training            # rebuild tables/plots only
python evaluate.py --checkpoint runs/<arm>/best.pt
python predict.py --checkpoint runs/<arm>/best.pt photo.jpg
```
