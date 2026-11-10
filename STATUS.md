# Project status — FER2013 / AffectNet baseline exercises

_Last updated: 2026-11-10_

Exercises are in `docs/insturtion.md`; the report is `report/lab-3.tex`.

## Code

| File | Purpose |
|---|---|
| `train.py` | train one model (mixup/CutMix, EMA, resumable), then test it |
| `evaluate.py` | test-set metrics, per-class report, confusion matrix |
| `predict.py` | predict the expression on your own photos |
| `experiments.py` | run the whole experiment matrix and write `RESULTS.md` and the comparison plots |
| `models.py`, `augment.py`, `utils.py`, `plots.py`, `data.py`, `config.py` | networks, GPU augmentation, EMA/metrics, plots, FER2013 + FER+ loading, options |
| `fer_experiments.ipynb` | Colab runner: clone this repo, run `experiments.py` |

## Fixed bug

Training intermittently produced a NaN loss on a GTX 1650 Ti (model predicted only "neutral").
Cause not proven; `cudnn.benchmark` + fp16 autotuning is the likely one. `cudnn.benchmark` is now off
and `train.py` skips any step with a non-finite loss. Verified on Colab (clean 8-epoch run).

## Experiment matrix (sprint profile, equal budget)

| Arm | Exercise |
|---|---|
| A1 ResNet-18, 128 px | 1 |
| A2 small stem, 48 px | 2 |
| A3 small stem + mixup/CutMix + EMA, 16 epochs | 3 |
| A4 EfficientNet-B0 + mixup + EMA | 3 (bonus) |
| B1 no pretraining | 4a |
| B2 AlexNet, 128 px | 4b |
| B3 ViT pretrained on expressions, frozen encoder | 4d |

Results: see `runs/RESULTS.md` (not committed; produced on Colab).

## Commands

```bash
python experiments.py --dry-run                  # print every command
python experiments.py --arms A --profile sprint  # half the matrix
python experiments.py --skip-training            # rebuild tables/plots only
python evaluate.py --checkpoint runs/<arm>/best.pt
python predict.py --checkpoint runs/<arm>/best.pt photo.jpg
```
