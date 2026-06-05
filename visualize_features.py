#!/usr/bin/env python3
"""
visualize_features.py — Feature map visualization for paper figure.

Passes a CIFAR-10 car image through Shape-ResNet-18 and baseline ResNet-18
under clean and corrupted (severity 4) conditions.

Output: 3-row x 4-column figure saved as PDF and PNG.

  Row 0: Shape-ResNet-18  | clean input
  Row 1: Shape-ResNet-18  | corrupted (severity 4)
  Row 2: Baseline ResNet-18 | corrupted (severity 4)

  Col 0: Input image
  Col 1: Stage 1 features  (layer1 / rgb.l1)
  Col 2: Stage 2 features  (layer2 / rgb.l2)
  Col 3: Stage 3 features  (layer3 fusion output / rgb.l3)

Usage:
    python visualize_features.py \
        --data_dir    /pvc/cifar10 \
        --ckpt_shape  /pvc/checkpoints/cifar-sweep/seed-42/shape_res18_cifar10.pt \
        --ckpt_base   /pvc/checkpoints/cifar-sweep/seed-42/baseline_res18_cifar10.pt \
        --corruption  gaussian_noise \
        --severity    4 \
        --car_idx     0 \
        --out_dir     /pvc/results/figures
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

from models import build_model

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--data_dir",   default="./data")
p.add_argument("--ckpt_shape", required=True)
p.add_argument("--ckpt_base",  required=True)
p.add_argument("--corruption", default="gaussian_noise",
               help="Corruption type from CIFAR-10-C")
p.add_argument("--severity",   type=int, default=4, choices=[1,2,3,4,5])
p.add_argument("--car_idx",    type=int, default=0,
               help="Which car image to use (0 = first car in test set)")
p.add_argument("--out_dir",    default="./figures")
args = p.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN   = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD    = torch.tensor([0.247,  0.243,  0.261 ]).view(3, 1, 1)
Path(args.out_dir).mkdir(parents=True, exist_ok=True)


# ── Image loading ─────────────────────────────────────────────────────────────



# ── Feature extraction ────────────────────────────────────────────────────────

def load_model(model_name: str, ckpt_path: str):
    model = build_model(model_name, num_classes=10, dataset="cifar10")
    state = torch.load(ckpt_path, map_location="cpu")
    sd    = state["model"] if "model" in state else state
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()


def extract_features(model, model_name: str, img_tensor: torch.Tensor) -> dict:
    """Register hooks, run forward pass, return feature maps."""
    features = {}
    hooks    = []

    def make_hook(key):
        def fn(module, inp, out):
            features[key] = out.detach().cpu()
        return fn

    if model_name == "shape_res18":
        hooks += [
            model.rgb.l1.register_forward_hook(make_hook("stage1")),
            model.rgb.l2.register_forward_hook(make_hook("stage2")),
            model.fusion.register_forward_hook(make_hook("stage3")),
        ]
    elif model_name == "baseline_res18":
        hooks += [
            model.model.layer1.register_forward_hook(make_hook("stage1")),
            model.model.layer2.register_forward_hook(make_hook("stage2")),
            model.model.layer3.register_forward_hook(make_hook("stage3")),
        ]

    with torch.no_grad():
        model(img_tensor.unsqueeze(0).to(DEVICE))

    for h in hooks:
        h.remove()
    return features


def feat_to_vis(feat: torch.Tensor) -> np.ndarray:
    """Mean across channels, normalize to [0,1]."""
    f = feat[0].mean(0).numpy()
    return (f - f.min()) / (f.max() - f.min() + 1e-8)


# ── Helpers for class-specific loading ────────────────────────────────────────

CIFAR10_CLASSES = {
    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3,
    "deer": 4,     "dog": 5,        "frog": 6, "horse": 7,
    "ship": 8,     "truck": 9,
}

def get_clean_image(data_dir: str, class_idx: int, img_idx: int = 0):
    ds = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=T.ToTensor())
    indices = [i for i, (_, lbl) in enumerate(ds) if lbl == class_idx]
    raw_tensor, _ = ds[indices[img_idx]]
    normalized    = (raw_tensor - MEAN) / STD
    raw_np        = (raw_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return normalized, raw_np

def get_corrupted_image(data_dir: str, corruption: str,
                        severity: int, class_idx: int, img_idx: int = 0):
    c_dir  = os.path.join(data_dir, "CIFAR-10-C")
    x_all  = np.load(os.path.join(c_dir, f"{corruption}.npy"))
    y_all  = np.load(os.path.join(c_dir, "labels.npy"))
    sev_x  = x_all[(severity - 1) * 10000 : severity * 10000]
    sev_y  = y_all[(severity - 1) * 10000 : severity * 10000]
    idx    = np.where(sev_y == class_idx)[0]
    raw_np = sev_x[idx[img_idx]]
    raw_t  = torch.from_numpy(raw_np).float().permute(2, 0, 1) / 255.0
    return (raw_t - MEAN) / STD, raw_np


# ── Load images ───────────────────────────────────────────────────────────────
print(f"Loading car image (idx={args.car_idx}) ...")
clean_tensor,   clean_np   = get_clean_image(args.data_dir, 1, args.car_idx)
corrupt_tensor, corrupt_np = get_corrupted_image(
    args.data_dir, args.corruption, args.severity, 1, args.car_idx)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading Shape-ResNet-18 ...")
shape_model = load_model("shape_res18", args.ckpt_shape)
print("Loading baseline ResNet-18 ...")
base_model  = load_model("baseline_res18", args.ckpt_base)

# ── Extract feature maps ──────────────────────────────────────────────────────
print("Extracting features ...")
feat_shape_clean = extract_features(shape_model, "shape_res18",    clean_tensor)
feat_shape_corr  = extract_features(shape_model, "shape_res18",    corrupt_tensor)
feat_base_clean  = extract_features(base_model,  "baseline_res18", clean_tensor)
feat_base_corr   = extract_features(base_model,  "baseline_res18", corrupt_tensor)

# ── Figure (4 rows × 4 cols) ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":      8,
    "axes.titlesize": 7,
})

ROW_LABELS = [
    "Shape-ResNet-18\n(clean)",
    "Shape-ResNet-18\n(corrupted, sev. 4)",
    "Baseline ResNet-18\n(clean)",
    "Baseline ResNet-18\n(corrupted, sev. 4)",
]
COL_LABELS = ["Input", "Stage 1", "Stage 2", "Stage 3"]
ROWS = [
    (clean_np,   feat_shape_clean),
    (corrupt_np, feat_shape_corr),
    (clean_np,   feat_base_clean),
    (corrupt_np, feat_base_corr),
]

fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.2))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

for r, (raw_np, feats) in enumerate(ROWS):
    ax = axes[r, 0]
    ax.imshow(raw_np)
    ax.set_xticks([]); ax.set_yticks([])
    if r == 0:
        ax.set_title(COL_LABELS[0], fontsize=8, pad=4)
    ax.set_ylabel(ROW_LABELS[r], fontsize=7, labelpad=4)

    for c, stage in enumerate(["stage1", "stage2", "stage3"], start=1):
        ax = axes[r, c]
        vis = feat_to_vis(feats[stage])
        ax.imshow(vis, cmap="inferno", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(COL_LABELS[c], fontsize=8, pad=4)

# Separator line between the two model groups (after row 1)
for c in range(4):
    axes[1, c].spines["bottom"].set_linewidth(1.2)
    axes[1, c].spines["bottom"].set_color("#555555")
# Thin borders between clean/corrupt within each group
for r in [0, 2]:
    for c in range(4):
        axes[r, c].spines["bottom"].set_linewidth(0.5)
        axes[r, c].spines["bottom"].set_color("#aaaaaa")

stem = os.path.join(args.out_dir, f"feature_vis_{args.corruption}_sev{args.severity}")
fig.savefig(f"{stem}.pdf", format="pdf", bbox_inches="tight", dpi=150)
fig.savefig(f"{stem}.png", format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"Saved {stem}.pdf and {stem}.png")


# ── Figure 2: Horse — Stage 1 across all severity levels ─────────────────────
# Layout: 3 rows × 6 cols (clean + severity 1-5)
# Row 0: input images
# Row 1: Shape-ResNet-18 stage 1 features
# Row 2: Baseline ResNet-18 layer 1 features

print("\nGenerating horse severity figure ...")
HORSE_CLASS = 7
HORSE_IDX   = 0   # first horse in test set

horse_clean_t, horse_clean_np = get_clean_image(args.data_dir, HORSE_CLASS, HORSE_IDX)

# Collect images and features for clean + severities 1-5
all_inputs   = [horse_clean_np]
feat_shape_s1 = [extract_features(shape_model, "shape_res18",    horse_clean_t)["stage1"]]
feat_base_l1  = [extract_features(base_model,  "baseline_res18", horse_clean_t)["stage1"]]

for sev in range(1, 6):
    h_t, h_np = get_corrupted_image(
        args.data_dir, args.corruption, sev, HORSE_CLASS, HORSE_IDX)
    all_inputs.append(h_np)
    feat_shape_s1.append(
        extract_features(shape_model, "shape_res18",    h_t)["stage1"])
    feat_base_l1.append(
        extract_features(base_model,  "baseline_res18", h_t)["stage1"])

COL_TITLES_HORSE = ["Clean", "Severity 1", "Severity 2",
                    "Severity 3", "Severity 4", "Severity 5"]
ROW_LABELS_HORSE = [
    "Input",
    "Shape-ResNet-18\n(Stage 1)",
    "Baseline ResNet-18\n(Layer 1)",
]

fig2, axes2 = plt.subplots(3, 6, figsize=(10.8, 4.5))
fig2.subplots_adjust(hspace=0.05, wspace=0.05)

for c in range(6):
    # Row 0: input image
    ax = axes2[0, c]
    ax.imshow(all_inputs[c])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(COL_TITLES_HORSE[c], fontsize=7, pad=3)
    if c == 0:
        ax.set_ylabel(ROW_LABELS_HORSE[0], fontsize=7, labelpad=4)

    # Row 1: Shape-ResNet-18 stage 1
    ax = axes2[1, c]
    ax.imshow(feat_to_vis(feat_shape_s1[c]), cmap="inferno", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if c == 0:
        ax.set_ylabel(ROW_LABELS_HORSE[1], fontsize=7, labelpad=4)

    # Row 2: Baseline layer 1
    ax = axes2[2, c]
    ax.imshow(feat_to_vis(feat_base_l1[c]), cmap="inferno", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    if c == 0:
        ax.set_ylabel(ROW_LABELS_HORSE[2], fontsize=7, labelpad=4)

# Thin row separators
for r in range(2):
    for c in range(6):
        axes2[r, c].spines["bottom"].set_linewidth(0.5)
        axes2[r, c].spines["bottom"].set_color("#aaaaaa")

stem2 = os.path.join(args.out_dir, f"horse_severity_{args.corruption}")
fig2.savefig(f"{stem2}.pdf", format="pdf", bbox_inches="tight", dpi=150)
fig2.savefig(f"{stem2}.png", format="png", bbox_inches="tight", dpi=150)
plt.close(fig2)
print(f"Saved {stem2}.pdf and {stem2}.png")
