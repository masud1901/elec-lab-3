"""Run the whole exercise matrix and build every table and figure the report needs.

This file is the single source of truth for the experiments, so the Colab notebook can stay unchanged:
edit ARMS here, push to GitHub, and the same notebook cell picks the change up.

    python experiments.py                          # sprint profile, every arm (~25-30 min on a T4)
    python experiments.py --profile full           # 25-50 epochs, higher-quality numbers
    python experiments.py --arms A                 # only the first half (for a groupmate running B)
    python experiments.py --arms A1_ex1_baseline_resnet18_128px
    python experiments.py --skip-training          # only rebuild the tables/plots from runs/
    python experiments.py --dry-run                # print the commands without running anything

Outputs (in --out-dir, default runs/):
    <arm>/...                     one folder per arm: checkpoints, history.csv, test_results.json, plots/
    results_summary.csv           one row per arm, sorted by macro-F1
    per_class_f1.csv              per-class F1 for every arm
    RESULTS.md                    the same table as markdown, ready to paste into the report
    learning_curves_all.png       train/val loss, val accuracy, val macro-F1 for every arm (exercise 4c)
    confusion_matrices_all.png    normalised confusion matrix per arm
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from config import DEFAULTS, get_config, run_name

# ----------------------------------------------------------------------------- the experiment matrix
# Each arm is one row of the report. `args` are ordinary train.py flags, so adding an experiment never
# needs a notebook change: add a dict entry here, push, re-run.
ARMS = {
    # ---- A: baseline, small images, improvements ------------------------------------------------
    "A1_ex1_baseline_resnet18_128px": dict(
        exercise="1", est=6,
        args=["--img-size", "128", "--batch-size", "128"]),
    "A2_ex2_small_stem_48px": dict(
        exercise="2", est=3,
        args=["--img-size", "48", "--small-stem", "--batch-size", "256"]),
    "A3_ex3_mixup_cutmix_ema": dict(
        exercise="3", est=6, epochs_mult=2,
        args=["--img-size", "48", "--small-stem", "--batch-size", "256",
              "--mixup", "0.2", "--cutmix", "1.0", "--ema-decay", "0.999",
              "--label-smoothing", "0.05", "--lr", "1e-3"]),
    "A4_bonus_efficientnet_b0": dict(
        exercise="3", est=5,
        args=["--model", "efficientnet_b0", "--img-size", "128", "--batch-size", "128",
              "--lr", "1e-3", "--mixup", "0.2", "--ema-decay", "0.999"]),
    # ---- B: pretraining -------------------------------------------------------------------------
    "B1_ex4a_no_pretrain": dict(
        exercise="4a", est=3,
        args=["--img-size", "48", "--small-stem", "--batch-size", "256", "--no-pretrained"]),
    "B2_ex4b_alexnet_128px": dict(
        exercise="4b", est=8,
        args=["--model", "alexnet", "--img-size", "128", "--batch-size", "128", "--lr", "5e-4"]),
    "B3_ex4d_vit_face": dict(
        exercise="4d", est=6,
        args=["--model", "vit_face", "--img-size", "112", "--batch-size", "64",
              "--freeze-encoder", "--lr", "1e-3"]),
}

PROFILES = {"sprint": 8, "full": 25}      # epochs per arm; arms with epochs_mult get a multiple
GROUPS = {"A": [n for n in ARMS if n.startswith("A")],
          "B": [n for n in ARMS if n.startswith("B")],
          "all": list(ARMS)}


# ----------------------------------------------------------------------------- running

def arm_argv(name, profile, out_dir, epochs=None):
    spec = ARMS[name]
    ep = epochs or PROFILES[profile] * spec.get("epochs_mult", 1)
    return list(spec["args"]) + ["--epochs", str(ep), "--out-dir", str(out_dir)]


def run_arm(name, argv, quiet=False):
    """Run one training job, streaming its epoch lines. Returns the run folder."""
    cfg = get_config(argv)
    run_dir = Path(cfg.out_dir) / run_name(cfg)
    if (run_dir / "test_results.json").exists():
        print(f"skip {name}: already finished ({run_dir.name})")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> {name}  [{ARMS[name]['exercise']}]  ~{ARMS[name]['est']} min on a T4")
    print(f"    {' '.join([sys.executable, '-u', 'train.py'] + argv)}", flush=True)
    t0 = time.time()
    with open(run_dir / "train_log.txt", "w") as log:
        proc = subprocess.Popen([sys.executable, "-u", "train.py"] + argv,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            s = line.replace("\r", " ").strip()
            if not quiet and (s.startswith("epoch") or "TEST" in s
                              or s.startswith(("Traceback", "RuntimeError", "Error"))):
                print(s, flush=True)
        proc.wait()

    minutes = (time.time() - t0) / 60
    timings = Path(cfg.out_dir) / "timings.json"
    done = json.loads(timings.read_text()) if timings.exists() else {}
    done[run_dir.name] = round(minutes, 1)
    timings.write_text(json.dumps(done, indent=2))
    print(f"done {name} in {minutes:.1f} min (exit {proc.returncode})")
    return run_dir


# ----------------------------------------------------------------------------- reporting

def collect(out_dir):
    out_dir = Path(out_dir)
    timings_path = out_dir / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    rows = []
    for f in sorted(out_dir.glob("*/test_results.json")):
        r = json.loads(f.read_text())
        c = json.loads((f.parent / "config.json").read_text())
        rows.append({
            "arm": f.parent.name,
            "exercise": next((ARMS[n]["exercise"] for n in ARMS if f.parent.name.startswith(n)), ""),
            "model": c.get("model"),
            "img": c.get("img_size"),
            "stem": "small" if c.get("small_stem") else "standard",
            "pretrained": c.get("pretrained"),
            "epochs": c.get("epochs"),
            "lr": c.get("lr"),
            "mixup": c.get("mixup"),
            "cutmix": c.get("cutmix"),
            "ema": c.get("ema_decay"),
            "accuracy": round(r["accuracy"], 4),
            "macro_f1": round(r["macro_f1"], 4),
            "best_epoch": r.get("epoch"),
            "minutes": timings.get(f.parent.name),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def _markdown_table(df, index=False):
    """Markdown table without needing tabulate."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        head = ([""] if index else []) + [str(c) for c in df.columns]
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for i, row in df.iterrows():
            cells = ([str(i)] if index else []) + ["" if pd.isna(v) else str(v) for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


def write_reports(out_dir, project_dir="."):
    out_dir, project_dir = Path(out_dir), Path(project_dir)
    table = collect(out_dir)
    if table.empty:
        print("no finished runs found in", out_dir)
        return table

    table.to_csv(out_dir / "results_summary.csv", index=False)

    per_class = {}
    for f in sorted(out_dir.glob("*/test_results.json")):
        per_class[f.parent.name] = json.loads(f.read_text())["per_class_f1"]
    pd.DataFrame(per_class).T.round(3).to_csv(out_dir / "per_class_f1.csv")

    cols = ["exercise", "arm", "model", "img", "stem", "pretrained", "epochs", "accuracy", "macro_f1", "minutes"]
    md = ["# Results", "",
          "Sorted by macro-F1 (the main score); accuracy is for comparison with published papers.", "",
          _markdown_table(table[cols]), "",
          "## Per-class F1", "",
          _markdown_table(pd.DataFrame(per_class).T.round(3), index=True)]
    (out_dir / "RESULTS.md").write_text("\n".join(md))

    print("\n" + table[cols].to_string(index=False))
    print(f"\nwrote {out_dir / 'results_summary.csv'}, {out_dir / 'per_class_f1.csv'}, {out_dir / 'RESULTS.md'}")
    plot_curves(out_dir)
    plot_confusions(out_dir)
    return table


def plot_curves(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = {d.name: pd.read_csv(d / "history.csv") for d in sorted(Path(out_dir).iterdir())
            if (d / "history.csv").exists()}
    if not runs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for metric, ax in zip(["train_loss", "val_loss", "accuracy", "macro_f1"], axes.ravel()):
        for name, h in runs.items():
            ax.plot(h["epoch"], h[metric], marker="o", markersize=3, label=name)
        ax.set_title(metric)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Training dynamics - every arm (exercise 4c)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "learning_curves_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", Path(out_dir) / "learning_curves_all.png")


def plot_confusions(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    runs = [(d.name, json.loads((d / "test_results.json").read_text()))
            for d in sorted(Path(out_dir).iterdir()) if (d / "test_results.json").exists()]
    if not runs:
        return
    n, cols = len(runs), min(3, len(runs))
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(5 * cols, 4.4 * -(-n // cols)), squeeze=False)
    for ax, (name, r) in zip(axes.ravel(), runs):
        cm = np.array(r["confusion_matrix"], dtype=float)
        ax.imshow(cm / np.clip(cm.sum(1, keepdims=True), 1, None), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(r["class_names"])))
        ax.set_xticklabels(r["class_names"], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(r["class_names"])))
        ax.set_yticklabels(r["class_names"], fontsize=7)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confusion_matrices_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", Path(out_dir) / "confusion_matrices_all.png")


# ----------------------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the FER2013 exercise matrix and build the report")
    parser.add_argument("--profile", choices=list(PROFILES), default="sprint",
                        help="epoch budget: sprint 8, full 25 (default: sprint)")
    parser.add_argument("--arms", default="all",
                        help="'all', 'A', 'B' or a comma-separated list of arm names (default: all)")
    parser.add_argument("--out-dir", default=DEFAULTS["out_dir"], help="where runs and reports go")
    parser.add_argument("--epochs", type=int, default=None, help="override the profile's epoch count")
    parser.add_argument("--skip-training", action="store_true", help="only rebuild the tables and plots")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = parser.parse_args(argv)

    if args.arms in GROUPS:
        names = GROUPS[args.arms]
    else:
        names = [n.strip() for n in args.arms.split(",") if n.strip()]
        unknown = [n for n in names if n not in ARMS]
        if unknown:
            parser.error(f"unknown arm(s): {unknown}. Choose from: {', '.join(ARMS)}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for name in names:
            print(name, "->", " ".join(["python", "train.py"] + arm_argv(name, args.profile, args.out_dir,
                                                                         args.epochs)))
        return

    if not args.skip_training:
        print(f"profile {args.profile}: {PROFILES[args.profile]} epochs per arm "
              f"(x2 for longer-schedule arms) | {len(names)} arm(s)")
        start = time.time()
        for name in names:
            run_arm(name, arm_argv(name, args.profile, args.out_dir, args.epochs))
        print(f"\ntraining finished in {(time.time() - start) / 60:.1f} min")

    write_reports(args.out_dir)


if __name__ == "__main__":
    main()
