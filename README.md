# FER2013 expression recognition: AffectNet-style baseline

Workshop 3 (Nov 10): reproduce the AffectNet paper's classification baseline on FER2013 with FER+ labels
(8 classes), then answer five exercises (`docs/insturtion.md`). The report is `report/lab-3.tex`.

## Results (test set, 8 epochs per arm on a Colab T4)

| Exercise | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| 1 | ResNet-18, 128 px (baseline) | 0.841 | 0.694 |
| 2 | ResNet-18, small stem, 48 px | 0.783 | 0.628 |
| 3 | small stem + mixup/CutMix + EMA (16 epochs) | 0.778 | 0.648 |
| 3 | EfficientNet-B0 + mixup + EMA, 128 px | 0.826 | 0.682 |
| 4a | ResNet-18, no pretraining | 0.742 | 0.557 |
| 4b | AlexNet, 128 px | 0.785 | 0.613 |
| 4d | ViT pretrained on expressions (frozen) | validation only: 0.697 | 0.501 |

Full Colab log: `logs/colab_sprint_log.txt`.

## Files

| File | Purpose |
|---|---|
| `train.py` | train one model (mixup/CutMix, EMA, resumable), then test it |
| `evaluate.py` | test metrics, per-class report, confusion matrix |
| `predict.py` | predict the expression on your own photos |
| `experiments.py` | run the whole experiment matrix, write `RESULTS.md` and comparison plots |
| `models.py`, `augment.py`, `utils.py`, `plots.py`, `data.py`, `config.py` | networks, GPU augmentation, EMA and metrics, plots, data loading, options |
| `fer_experiments.ipynb` | Colab runner: clone this repo and run `experiments.py` |

## Usage

```bash
pip install -r requirements.txt
python train.py                                   # exercise 1 baseline
python experiments.py --profile sprint --arms all # every arm (~20 min on a T4)
python evaluate.py --checkpoint runs/<run>/best.pt
python predict.py --checkpoint runs/<run>/best.pt photo.jpg
```

## Notes

- Training intermittently produced NaN losses on a GTX 1650 Ti. `cudnn.benchmark` is now off and
  `train.py` skips any step with a non-finite loss.
- The ViT arm originally crashed in `evaluate.py` (pooler layer mismatch); fixed in `models.py`.

## Acknowledgements

I used Claude (Claude Code) and OpenCode as coding assistants for parts of this code, the experiments and
the report. I ran the experiments and checked the results myself.
