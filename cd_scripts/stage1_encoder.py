import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageEnhance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def normalize_numeric_name(value, width: int | None = None) -> str:
    s = str(value).strip()
    if not s:
        return s
    p = Path(s)
    stem = p.stem.strip()
    suffix = p.suffix.strip().lower()
    if stem.isdigit():
        stem = stem.zfill(width or len(stem))
        return f"{stem}{suffix}"
    return s

def normalize_patient_id(pid) -> str:
    return normalize_numeric_name(pid, width=8)

def normalize_image_name(image_name) -> str:
    s = str(image_name).strip()
    if not s:
        return s
    p = Path(s)
    stem = p.stem.strip()
    suffix = p.suffix.strip().lower()
    if stem.isdigit():
        stem = stem.zfill(6)
        return f"{stem}{suffix}"
    return s

def build_image_candidates(image_dir: str | Path, patient_id: str, image_name: str):
    image_dir = Path(image_dir)
    pid_raw = str(patient_id).strip()
    pid_norm = normalize_patient_id(pid_raw)
    img_raw = str(image_name).strip()
    img_norm = normalize_image_name(img_raw)

    candidate_patient_dirs = []
    for x in [pid_norm, pid_raw]:
        if x and x not in candidate_patient_dirs:
            candidate_patient_dirs.append(x)

    candidate_image_names = []
    for x in [img_norm, img_raw]:
        if x and x not in candidate_image_names:
            candidate_image_names.append(x)

    candidates = []
    for pid_dir in candidate_patient_dirs:
        for img_name in candidate_image_names:
            c = image_dir / pid_dir / img_name
            if c not in candidates:
                candidates.append(c)
            if not c.suffix:
                for ext in IMAGE_EXTENSIONS:
                    cc = c.with_suffix(ext)
                    if cc not in candidates:
                        candidates.append(cc)
            else:
                base = c.with_suffix("")
                for ext in IMAGE_EXTENSIONS:
                    cc = base.with_suffix(ext)
                    if cc not in candidates:
                        candidates.append(cc)
    return candidates

def resolve_image_path(image_dir: str | Path, patient_id: str, image_name: str):
    for c in build_image_candidates(image_dir, patient_id, image_name):
        if c.exists():
            return c
    return None

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def configure_gpu(cfg: dict):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(cfg.get("allow_tf32", True))
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda" else "cpu")

def ensure_dirs(cfg: dict):
    out = Path(cfg["output_dir"])
    (out / "stage1").mkdir(parents=True, exist_ok=True)
    return out

def amp_enabled(cfg: dict, device: torch.device) -> bool:
    return bool(cfg.get("use_amp", True) and device.type == "cuda")

def make_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)

def autocast_ctx(enabled: bool):
    return torch.amp.autocast("cuda", enabled=enabled)

def normalize_batch_on_device(x: torch.Tensor, device: torch.device, enabled: bool = True) -> torch.Tensor:
    if not enabled:
        return x
    mean = IMAGENET_MEAN.to(device=device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=device, dtype=x.dtype)
    return (x - mean) / std

def build_loader_kwargs(num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int | None):
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers and num_workers > 0),
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs

def binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pos_rate = float(y_pred.mean()) if len(y_pred) else np.nan
    return {"auc": float(auc), "acc": float(acc), "f1": float(f1), "pos_rate": pos_rate}


class CDImageDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform):
        df = df.reset_index(drop=True)
        self.img_paths = df["img_path"].astype(str).tolist()
        self.labels = df["annotated_label"].astype(int).tolist()
        self.patient_ids = df["patient_id"].astype(str).tolist()
        self.image_names = df["image"].astype(str).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = Path(self.img_paths[idx])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to open image: {img_path}") from e
        img = self.transform(img)
        return {
            "image": img,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "patient_id": self.patient_ids[idx],
            "image_name": self.image_names[idx],
        }

def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.06, 0.06, 0.04, 0.02),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return train_tf, eval_tf


def attach_and_filter_existing_images(img_df: pd.DataFrame, image_dir: str, cache_dir: Path):
    df = img_df.copy()
    df["patient_id_raw"] = df["patient_id"].astype(str).str.strip()
    df["image_raw"] = df["image"].astype(str).str.strip()
    df["patient_id"] = df["patient_id_raw"].map(normalize_patient_id)
    df["image"] = df["image_raw"].map(normalize_image_name)

    resolved = [resolve_image_path(image_dir, pid_raw, img_raw) for pid_raw, img_raw in zip(df["patient_id_raw"], df["image_raw"])]
    df["img_path"] = [str(p) if p is not None else "" for p in resolved]
    exists_mask = df["img_path"] != ""
    missing_df = df.loc[~exists_mask, ["patient_id_raw", "patient_id", "image_raw", "image", "annotated_label", "img_path"]].copy()
    kept_df = df.loc[exists_mask].reset_index(drop=True)

    missing_csv = cache_dir / "stage1_missing_images.csv"
    missing_df.to_csv(missing_csv, index=False)

    patient_total = df.groupby("patient_id").size().rename("n_total")
    patient_kept = kept_df.groupby("patient_id").size().rename("n_kept")
    patient_stat = pd.concat([patient_total, patient_kept], axis=1).fillna(0)
    patient_stat["n_total"] = patient_stat["n_total"].astype(int)
    patient_stat["n_kept"] = patient_stat["n_kept"].astype(int)
    valid_patients = set(patient_stat.index[patient_stat["n_kept"] > 0].tolist())

    print(f"[Stage1] images before filtering: {len(df)}")
    print(f"[Stage1] images after filtering : {len(kept_df)}")
    print(f"[Stage1] missing image rows      : {len(missing_df)}")
    print(f"[Stage1] valid patients         : {len(valid_patients)}")
    print(f"[Stage1] missing image log saved : {missing_csv}")

    return kept_df, missing_df, valid_patients


def load_gastronet_backbone_weights(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["teacher"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            k = k[len("backbone."):]
        if k == "pos_embed" and tuple(v.shape) == (1, 577, 768):
            v = v[:, 1:, :]
        if k == "register_tokens":
            k = "reg_token"
        if k.startswith("dino_head") or k == "mask_token":
            continue
        clean_state_dict[k] = v
    return clean_state_dict

class GastroNetStage1Classifier(nn.Module):

    def __init__(self, backbone_name: str, ckpt_path: str, image_size: int, abnormal_logit_bias: float = 0.0):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=False, img_size=image_size, num_classes=0)
        msg = self.encoder.load_state_dict(load_gastronet_backbone_weights(ckpt_path), strict=False)
        if len(msg.missing_keys) > 0:
            raise RuntimeError(f"Missing keys when loading GastroNet backbone: {msg.missing_keys}")

        self.classifier = nn.Linear(768, 2)

        if abnormal_logit_bias != 0:
            with torch.no_grad():
                self.classifier.bias[1] += abnormal_logit_bias

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def extract_cls(self, x):
        tokens = self.encoder.forward_features(x)
        return tokens[:, 0, :]

    def forward(self, x):
        cls_feat = self.extract_cls(x)
        return self.classifier(cls_feat)

class GastroNetFeatureEncoder(nn.Module):

    def __init__(self, backbone_name: str, image_size: int):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=False, img_size=image_size, num_classes=0)

    def forward(self, x):
        tokens = self.encoder.forward_features(x)
        return tokens[:, 0, :]


def evaluate(model, loader, device, use_amp: bool, channels_last: bool, gpu_normalize: bool):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            x = normalize_batch_on_device(x, device, enabled=gpu_normalize)
            with autocast_ctx(use_amp):
                logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())
    return binary_metrics(labels, probs)

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp: bool, max_grad_norm: float, channels_last: bool, gpu_normalize: bool):
    model.train()
    total_loss = 0.0
    total_samples = 0
    data_time_total = 0.0
    compute_time_total = 0.0
    batch_count = 0
    iter_start = time.perf_counter()

    for batch in loader:
        data_ready = time.perf_counter()
        data_time_total += data_ready - iter_start

        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x = normalize_batch_on_device(x, device, enabled=gpu_normalize)

        optimizer.zero_grad(set_to_none=True)
        comp_start = time.perf_counter()

        with autocast_ctx(use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        compute_time_total += time.perf_counter() - comp_start
        total_loss += float(loss.item())
        total_samples += int(y.size(0))
        batch_count += 1
        iter_start = time.perf_counter()

    stats = {
        "loss": total_loss / max(batch_count, 1),
        "samples": total_samples,
        "data_time": data_time_total / max(batch_count, 1),
        "compute_time": compute_time_total / max(batch_count, 1),
        "throughput": total_samples / max(data_time_total + compute_time_total, 1e-6),
    }
    return stats


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300

def plot_roc_curve_pub(y_true, y_prob, out_path, color='#E64B35'):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Abnormal Image Classification', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='svg')
    plt.savefig(str(out_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] ROC curve: {out_path}")
    return auc

def plot_calibration_curve_pub(y_true, y_prob, out_path, color='#E64B35', label='Model'):
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy='uniform'
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(mean_predicted_value, fraction_of_positives, 's-',
            color=color, linewidth=2, markersize=8, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='svg')
    plt.savefig(str(out_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Calibration curve: {out_path}")

def plot_confusion_matrix_pub(y_true, y_pred, out_path, color='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=axes[0],
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                cbar_kws={'shrink': 0.8})
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('Actual', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap=color, ax=axes[1],
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'],
                cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='svg')
    plt.savefig(str(out_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Confusion matrix: {out_path}")

def plot_attention_heatmap(model, image_paths, labels, out_dir, device, image_size=336, prefix="sample"):
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for idx, (img_path, label) in enumerate(zip(image_paths[:50], labels[:50])):
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((image_size, image_size))
            img_tensor = transform(img_resized).unsqueeze(0).to(device)
            img_tensor.requires_grad_(True)

            features = model["encoder"].forward_features(img_tensor)
            cls_feat = features[:, 0, :]
            logits = model["classifier"](cls_feat)
            pred_class = logits.argmax(dim=1).item()
            pred_prob = torch.softmax(logits, dim=1)[0, pred_class].item()

            model.zero_grad()
            one_hot = torch.zeros_like(logits)
            one_hot[0, pred_class] = 1
            logits.backward(gradient=one_hot, retain_graph=True)

            grad = img_tensor.grad
            if grad is not None:
                grad_np = grad.squeeze().cpu().numpy()
                if grad_np.ndim == 3:
                    saliency = np.abs(grad_np).mean(axis=0)
                else:
                    saliency = np.abs(grad_np)
                saliency = saliency / (saliency.max() + 1e-8)
                from scipy.ndimage import gaussian_filter
                saliency = gaussian_filter(saliency, sigma=5)
                saliency = saliency / (saliency.max() + 1e-8)
            else:
                saliency = np.ones((image_size, image_size)) * 0.5

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_resized)
            axes[0].set_title(f'Original\nLabel: {"Abnormal" if label else "Normal"} | Pred: {pred_class} ({pred_prob:.2f})',
                             fontsize=11, fontweight='bold')
            axes[0].axis('off')

            im = axes[1].imshow(saliency, cmap='turbo', interpolation='bilinear')
            axes[1].set_title('Attention Map\n(Gradient Sensitivity)', fontsize=11, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, shrink=0.8)

            axes[2].imshow(img_resized)
            axes[2].imshow(saliency, cmap='turbo', alpha=0.5, interpolation='bilinear')
            axes[2].set_title('Overlay', fontsize=11, fontweight='bold')
            axes[2].axis('off')

            plt.tight_layout()
            out_path = Path(out_dir) / f'{prefix}_{idx+1:02d}.svg'
            plt.savefig(out_path, dpi=300, bbox_inches='tight', format='svg')
            plt.savefig(str(out_path).replace('.svg', '.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  [Saved] {out_path}")

            model.zero_grad()
            img_tensor.grad = None

        except Exception as e:
            print(f"  [WARN] Failed to process image {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

class ImageAbnormalDataset(Dataset):

    def __init__(self, image_df, image_dir, transform=None, resolve_func=None):
        self.image_df = image_df.reset_index(drop=True)
        self.image_dir = str(image_dir)
        self.transform = transform
        self.resolve_func = resolve_func

        if "patient_id" in self.image_df.columns:
            self.image_df["patient_id_norm"] = self.image_df["patient_id"].astype(str).map(normalize_patient_id)
        if "image" in self.image_df.columns:
            self.image_df["image_norm"] = self.image_df["image"].astype(str).map(normalize_image_name)

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        row = self.image_df.iloc[idx]

        if self.resolve_func:
            patient_id_raw = str(row.get("patient_id", ""))
            image_raw = str(row.get("image", ""))
            img_path = self.resolve_func(self.image_dir, patient_id_raw, image_raw)
        else:
            img_path = row.get("img_path", "")

        try:
            if img_path and Path(img_path).exists():
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.new('RGB', (336, 336), (128, 128, 128))
        except Exception:
            img = Image.new('RGB', (336, 336), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        label = int(row["annotated_label"])
        patient_id = str(row.get("patient_id", f"img_{idx}"))

        return {
            "image": img,
            "label": label,
            "patient_id": patient_id,
            "image_path": str(img_path) if img_path else ""
        }

def run_comprehensive_evaluation(model, test_df, image_dir, device, image_size, abnormal_logit_bias, output_dir):
    print("\n" + "=" * 80)
    print("[Stage1] 运行综合评估...")
    print("=" * 80)

    eval_dir = Path(output_dir) / "stage1" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageAbnormalDataset(
        test_df, str(image_dir),
        transform=test_transform,
        resolve_func=resolve_image_path
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=48,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("[Evaluating] Running inference on test set...")
    all_probs = []
    all_labels = []
    all_patient_ids = []

    use_amp = torch.cuda.is_available()
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].numpy()
            patient_ids = batch["patient_id"]

            with torch.amp.autocast("cuda" if use_amp else "cpu"):
                cls_feat = model["encoder"].forward_features(images)
                cls_feat = cls_feat[:, 0, :]
                logits = model["classifier"](cls_feat)
                logits_calibrated = logits.clone()
                logits_calibrated[:, 1] -= abnormal_logit_bias

            probs = torch.softmax(logits_calibrated, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
            all_patient_ids.extend(patient_ids)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    print("\n[Results] Test Set Performance:")
    print("-" * 50)

    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs > 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    precision = precision_score(all_labels, preds)
    recall = recall_score(all_labels, preds)

    print(f"  AUC:       {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    cm = confusion_matrix(all_labels, preds)
    print(f"\n  Confusion Matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    print("\n[Calibration] Applying temperature scaling...")
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            with torch.amp.autocast("cuda" if use_amp else "cpu"):
                cls_feat = model["encoder"].forward_features(images)
                cls_feat = cls_feat[:, 0, :]
                logits = model["classifier"](cls_feat)
                logits_calibrated = logits.clone()
                logits_calibrated[:, 1] -= abnormal_logit_bias
            all_logits.append(logits_calibrated.cpu())
    all_logits = torch.cat(all_logits, dim=0)
    all_labels_tensor = torch.tensor(all_labels)

    def nll(logits, labels, T):
        probs = torch.softmax(logits / T, dim=1)
        return F.cross_entropy(torch.log(probs + 1e-8), labels)

    best_T = 1.0
    best_nll = float('inf')
    for T in np.linspace(0.5, 3.0, 50):
        nll_val = nll(all_logits, all_labels_tensor, T).item()
        if nll_val < best_nll:
            best_nll = nll_val
            best_T = T

    print(f"  [Calibration] Optimal temperature: {best_T:.3f}")
    calibrated_probs = torch.softmax(all_logits / best_T, dim=1)[:, 1].numpy()

    print("\n[Visualization] Generating plots...")

    plot_roc_curve_pub(all_labels, all_probs, eval_dir / "roc_curve.svg")

    plot_calibration_curve_pub(all_labels, calibrated_probs, eval_dir / "calibration_curve.svg",
                               color='#E64B35')
    plot_calibration_curve_pub(all_labels, all_probs, eval_dir / "calibration_curve_original.svg",
                               color='#999999')

    plot_confusion_matrix_pub(all_labels, preds, eval_dir / "confusion_matrix.svg")

    print("\n[Visualization] Generating attention heatmaps (50 normal + 50 abnormal)...")
    normal_samples = test_df[test_df["annotated_label"] == 0].sample(n=min(50, len(test_df[test_df["annotated_label"] == 0])), random_state=42)
    abnormal_samples = test_df[test_df["annotated_label"] == 1].sample(n=min(50, len(test_df[test_df["annotated_label"] == 1])), random_state=42)

    normal_paths = [resolve_image_path(image_dir, pid, img)
                   for pid, img in zip(normal_samples["patient_id"], normal_samples["image"])]
    normal_labels = normal_samples["annotated_label"].values
    plot_attention_heatmap(model, normal_paths, normal_labels, eval_dir / "attention_heatmaps" / "normal", device, image_size, prefix="normal")

    abnormal_paths = [resolve_image_path(image_dir, pid, img)
                     for pid, img in zip(abnormal_samples["patient_id"], abnormal_samples["image"])]
    abnormal_labels = abnormal_samples["annotated_label"].values
    plot_attention_heatmap(model, abnormal_paths, abnormal_labels, eval_dir / "attention_heatmaps" / "abnormal", device, image_size, prefix="abnormal")

    report = classification_report(all_labels, preds, target_names=['Normal', 'Abnormal'])
    report_path = eval_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Stage1 Evaluation (CD)\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Test Set Metrics:\n")
        f.write(f"  AUC:       {auc:.4f}\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n\n")
        f.write(f"Classification Report:\n{report}\n")

    print(f"  [Saved] Classification report: {report_path}")

    results = {
        "test_metrics": {
            "auc": float(auc),
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        },
        "calibration": {
            "temperature": float(best_T),
            "original_nll": float(nll(all_logits, all_labels_tensor, 1.0).item()),
            "calibrated_nll": float(best_nll),
        },
        "confusion_matrix": cm.tolist(),
        "n_test_samples": len(test_df),
        "evaluation_time": datetime.now().isoformat(),
    }

    results_path = eval_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  [Saved] Results: {results_path}")

    print("\n" + "=" * 80)
    print("[Stage1] 综合评估完成!")
    print(f"[Stage1] 可视化结果保存在: {eval_dir}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage1: GastroNet-5M Encoder Pretraining (CD Version)")
    parser.add_argument("--config", required=True, help="配置文件路径")
    args = parser.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    out_dir = ensure_dirs(cfg)
    stage1_dir = out_dir / "stage1"
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg = cfg["stage1"]

    seed_everything(cfg["seed"])

    device = configure_gpu(cfg)
    use_amp = amp_enabled(cfg, device)
    gpu_normalize = bool(stage_cfg.get("gpu_normalize", True))

    print("=" * 80)
    print("Stage 1: GastroNet-5M Encoder预训练 - CD异常图像分类")
    print("=" * 80)
    print(f"[Stage1] device={device} | cuda_available={torch.cuda.is_available()} | use_amp={use_amp}")
    if device.type == "cuda":
        try:
            print(f"[Stage1] gpu={torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    img_df = pd.read_csv(cfg["img_csv"])
    img_df = img_df[img_df["annotated_label"].isin([0, 1])].copy().reset_index(drop=True)
    patient_df = pd.read_csv(cfg["pat_csv"])
    patient_df = patient_df.dropna(subset=["remission", "healing"]).copy().reset_index(drop=True)
    patient_df["patient_id_raw"] = patient_df["patient_id"].astype(str).str.strip()
    patient_df["patient_id"] = patient_df["patient_id_raw"].map(normalize_patient_id)

    n_patient_total = patient_df["patient_id"].nunique()

    img_df, missing_df, valid_patients = attach_and_filter_existing_images(img_df, cfg["image_dir"], cache_dir)
    patient_ids = set(patient_df["patient_id"].tolist())
    valid_joint_patients = sorted(list(patient_ids & valid_patients))

    print(f"[Stage1] patients in pat_csv after label filtering : {n_patient_total}")
    print(f"[Stage1] patients with >=1 resolved image          : {len(valid_patients)}")
    print(f"[Stage1] patients after ID alignment intersection : {len(valid_joint_patients)}")

    patient_df = patient_df[patient_df["patient_id"].isin(valid_joint_patients)].reset_index(drop=True)
    img_df = img_df[img_df["patient_id"].isin(valid_joint_patients)].reset_index(drop=True)

    if len(patient_df) < 8:
        raise RuntimeError(f"Too few valid patients after filtering missing images: {len(patient_df)}")
    if img_df.empty:
        raise RuntimeError("No valid image rows remained after filtering missing files.")

    encoder_ckpt_path = stage1_dir / stage_cfg["encoder_ckpt_name"]

    if encoder_ckpt_path.exists():
        print("\n" + "=" * 80)
        print("[Stage1] 检测到已保存的encoder模型，跳过训练，直接进行综合评估!")
        print(f"[Stage1] encoder_ckpt_path: {encoder_ckpt_path}")
        print("=" * 80 + "\n")

        ckpt = torch.load(encoder_ckpt_path, map_location="cpu")
        cfg_stored = ckpt.get("config", {})
        backbone_name = cfg_stored.get("stage1", {}).get("backbone_name", stage_cfg["backbone_name"])
        image_size = cfg_stored.get("stage1", {}).get("image_size", stage_cfg["image_size"])
        abnormal_logit_bias = cfg_stored.get("stage1", {}).get("abnormal_logit_bias", stage_cfg.get("abnormal_logit_bias", 0.0))

        model = GastroNetStage1Classifier(
            backbone_name, cfg["gastronet_ckpt"], image_size,
            abnormal_logit_bias=abnormal_logit_bias
        )
        model.load_state_dict(ckpt["classifier_state_dict"], strict=False)
        model.to(device)
        model.eval()

        split_seed = int(cfg["train_test_split_seed"])
        strat = patient_df["remission"].astype(int).tolist()
        tr_pid, te_pid = train_test_split(
            patient_df["patient_id"].astype(str).tolist(),
            test_size=cfg["test_size"],
            random_state=split_seed,
            stratify=strat,
        )
        test_img = img_df[img_df["patient_id"].isin(te_pid)].reset_index(drop=True)

        run_comprehensive_evaluation(
            model, test_img, cfg["image_dir"], device, image_size,
            abnormal_logit_bias, out_dir
        )
        return
    img_df = img_df[img_df["patient_id"].isin(valid_joint_patients)].reset_index(drop=True)

    if len(patient_df) < 8:
        raise RuntimeError(f"Too few valid patients after filtering missing images: {len(patient_df)}")
    if img_df.empty:
        raise RuntimeError("No valid image rows remained after filtering missing files.")

    split_seed = int(cfg["train_test_split_seed"])
    strat = patient_df["remission"].astype(int).tolist()
    tr_pid, te_pid = train_test_split(
        patient_df["patient_id"].astype(str).tolist(),
        test_size=cfg["test_size"],
        random_state=split_seed,
        stratify=strat,
    )
    split_info = {"train_patient_ids": tr_pid, "test_patient_ids": te_pid}
    with open(cache_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    train_img = img_df[img_df["patient_id"].isin(tr_pid)].reset_index(drop=True)
    test_img = img_df[img_df["patient_id"].isin(te_pid)].reset_index(drop=True)
    img_strat = train_img["annotated_label"].astype(int)
    try:
        trn_img, val_img = train_test_split(train_img, test_size=0.15, random_state=split_seed, stratify=img_strat)
    except ValueError:
        trn_img, val_img = train_test_split(train_img, test_size=0.15, random_state=split_seed, stratify=None)
        print("[Stage1][WARN] train/val image stratified split failed; fell back to unstratified split.")

    train_tf, eval_tf = build_transforms(stage_cfg["image_size"])
    ds_tr = CDImageDataset(trn_img, train_tf)
    ds_va = CDImageDataset(val_img, eval_tf)
    ds_te = CDImageDataset(test_img, eval_tf)

    stage1_workers = int(stage_cfg.get("num_workers", cfg.get("num_workers", 4)))
    stage1_prefetch = int(stage_cfg.get("prefetch_factor", cfg.get("prefetch_factor", 2))) if stage1_workers > 0 else None
    loader_kwargs = build_loader_kwargs(stage1_workers, cfg["pin_memory"], cfg["persistent_workers"], stage1_prefetch)
    dl_tr = DataLoader(ds_tr, batch_size=stage_cfg["batch_size"], shuffle=True, **loader_kwargs)
    dl_va = DataLoader(ds_va, batch_size=stage_cfg["batch_size"], shuffle=False, **loader_kwargs)
    dl_te = DataLoader(ds_te, batch_size=stage_cfg["batch_size"], shuffle=False, **loader_kwargs)

    model = GastroNetStage1Classifier(
        stage_cfg["backbone_name"],
        cfg["gastronet_ckpt"],
        stage_cfg["image_size"],
        abnormal_logit_bias=stage_cfg.get("abnormal_logit_bias", 0.0)
    )
    model.freeze_backbone()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())

    print("\n" + "=" * 80)
    print("Stage1 模型初始化信息 (论文用)")
    print("=" * 80)
    print(f"[Stage1] Backbone: {stage_cfg['backbone_name']}")
    print(f"[Stage1] Image Size: {stage_cfg['image_size']}")
    print(f"[Stage1] 总参数量: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"[Stage1] Encoder参数量: {encoder_params:,} ({encoder_params / 1e6:.2f} M)")
    print(f"[Stage1] Classifier参数量: {classifier_params:,} ({classifier_params / 1e6:.2f} M)")
    print(f"[Stage1] 冻结状态可训练参数量: {trainable_params_frozen:,} ({trainable_params_frozen / 1e6:.2f} M)")
    print(f"[Stage1] 初始状态: Backbone冻结 (冻结 {stage_cfg.get('freeze_backbone_epochs', 3)} 个epoch)")
    print("=" * 80 + "\n")

    if cfg.get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)

    n_pos = max(int((trn_img["annotated_label"] == 1).sum()), 1)
    n_neg = max(int((trn_img["annotated_label"] == 0).sum()), 1)
    class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=stage_cfg.get("label_smoothing", 0.0))

    optimizer = torch.optim.AdamW([
        {"params": list(model.classifier.parameters()), "lr": stage_cfg["head_lr"]},
        {"params": list(model.encoder.parameters()), "lr": stage_cfg["lr"]},
    ], weight_decay=stage_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_cfg["epochs"])
    scaler = make_scaler(use_amp)

    best_score = -1.0
    best_epoch = -1
    no_improve = 0
    history = []

    print("\n" + "=" * 80)
    print("Stage1 开始训练...")
    print("=" * 80 + "\n")

    for epoch in range(1, stage_cfg["epochs"] + 1):
        if epoch == stage_cfg.get("freeze_backbone_epochs", 3) + 1:
            model.unfreeze_all()
            trainable_params_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n[Stage1] backbone unfrozen at epoch {epoch}")
            print(f"[Stage1] 解冻后全部可训练参数量: {trainable_params_unfrozen:,} ({trainable_params_unfrozen / 1e6:.2f} M)")

        train_stats = train_one_epoch(
            model, dl_tr, optimizer, criterion, scaler, device,
            use_amp=use_amp,
            max_grad_norm=stage_cfg.get("max_grad_norm", 1.0),
            channels_last=cfg.get("channels_last", True),
            gpu_normalize=gpu_normalize,
        )
        scheduler.step()
        val_metrics = evaluate(model, dl_va, device, use_amp=use_amp, channels_last=cfg.get("channels_last", True), gpu_normalize=gpu_normalize)

        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            **val_metrics,
            **{k: train_stats[k] for k in ["data_time", "compute_time", "throughput"]}
        })

        score = val_metrics["auc"] + 0.2 * val_metrics["f1"]
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % stage_cfg["eval_every"] == 0:
            gpu_mem = 0.0
            if device.type == "cuda":
                gpu_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                torch.cuda.reset_peak_memory_stats(device)
            print(
                f"[Stage1] epoch={epoch:03d} loss={train_stats['loss']:.4f} val_auc={val_metrics['auc']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} pos_rate={val_metrics['pos_rate']:.4f} "
                f"data_t={train_stats['data_time']:.3f}s compute_t={train_stats['compute_time']:.3f}s "
                f"samples_s={train_stats['throughput']:.1f} gpu_mem={gpu_mem:.2f}GB"
            )

        if epoch >= stage_cfg["min_epochs"] and no_improve >= stage_cfg["early_stop_patience"]:
            print(f"[Stage1] Early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break

    pd.DataFrame(history).to_csv(stage1_dir / "stage1_training_history.csv", index=False)

    print("\n" + "=" * 80)
    print(f"[Stage1] 第一阶段完成: 确定最佳epoch = {best_epoch}")
    print("[Stage1] 开始第二阶段：用全量训练数据重训练...")
    print("=" * 80)

    model = GastroNetStage1Classifier(
        stage_cfg["backbone_name"],
        cfg["gastronet_ckpt"],
        stage_cfg["image_size"],
        abnormal_logit_bias=stage_cfg.get("abnormal_logit_bias", 0.0)
    )
    model.to(device)

    full_train_df = pd.concat([trn_img, val_img], ignore_index=True)
    ds_full = CDImageDataset(full_train_df, train_tf)
    dl_full = DataLoader(ds_full, batch_size=stage_cfg["batch_size"], shuffle=True, **loader_kwargs)

    full_optimizer = torch.optim.AdamW([
        {"params": list(model.classifier.parameters()), "lr": stage_cfg["head_lr"]},
        {"params": list(model.encoder.parameters()), "lr": stage_cfg["lr"]},
    ], weight_decay=stage_cfg["weight_decay"])
    full_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(full_optimizer, T_max=best_epoch)

    n_pos = max(int((full_train_df["annotated_label"] == 1).sum()), 1)
    n_neg = max(int((full_train_df["annotated_label"] == 0).sum()), 1)
    full_class_weights = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=device)
    full_criterion = nn.CrossEntropyLoss(weight=full_class_weights, label_smoothing=stage_cfg.get("label_smoothing", 0.0))

    print(f"[Stage1] 开始全量数据重训练 (1-{best_epoch} epochs)...")
    for epoch in range(1, best_epoch + 1):
        if epoch == stage_cfg.get("freeze_backbone_epochs", 3) + 1:
            model.unfreeze_backbone()
            full_optimizer.param_groups[1]["lr"] = stage_cfg["lr"]
            print(f"[Stage1] backbone unfrozen at epoch {epoch}")

        full_stats = train_one_epoch(
            model, dl_full, full_optimizer, full_criterion, scaler, device,
            use_amp=use_amp, max_grad_norm=stage_cfg.get("max_grad_norm", 1.0),
            channels_last=cfg.get("channels_last", True), gpu_normalize=gpu_normalize
        )
        full_scheduler.step()

        if epoch == 1 or epoch % stage_cfg["eval_every"] == 0 or epoch == best_epoch:
            print(
                f"[Stage1] Full-train epoch={epoch:03d} loss={full_stats['loss']:.4f} "
                f"samples_s={full_stats['throughput']:.1f}"
            )

    print("\n" + "=" * 80)
    print("[Stage1] 第二阶段完成，保存最终模型...")
    print("=" * 80)

    encoder_wrapper = GastroNetFeatureEncoder(stage_cfg["backbone_name"], stage_cfg["image_size"])
    encoder_state = encoder_wrapper.state_dict()
    current_state = model.encoder.state_dict()
    remapped_state = {f"encoder.{k}": v.detach().cpu().clone() for k, v in current_state.items()}
    for k in encoder_state.keys():
        encoder_state[k] = remapped_state[k]

    torch.save({
        "epoch": best_epoch,
        "best_val_epoch": best_epoch,
        "val_metrics": history[best_epoch - 1] if best_epoch <= len(history) else history[-1],
        "classifier_state_dict": model.state_dict(),
        "encoder_state_dict": encoder_state,
        "config": cfg,
    }, encoder_ckpt_path)
    print(f"[Stage1] .pt模型已保存: {encoder_ckpt_path}")

    try:
        from safetensors.torch import save_file as safe_save_file

        hf_encoder_dir = stage1_dir / "hf_encoder_format"
        hf_encoder_dir.mkdir(parents=True, exist_ok=True)

        clean_encoder_state = {}
        for k, v in current_state.items():
            clean_k = k.replace("encoder.", "")
            tensor_v = v.detach().cpu().clone()
            if not tensor_v.is_contiguous():
                tensor_v = tensor_v.contiguous()
            clean_encoder_state[clean_k] = tensor_v

        hf_safetensors_path = hf_encoder_dir / "model.safetensors"
        safe_save_file(clean_encoder_state, str(hf_safetensors_path))

        hf_config = {
            "model_type": "vit",
            "backbone_name": stage_cfg["backbone_name"],
            "image_size": stage_cfg["image_size"],
            "feature_dim": stage_cfg["feature_dim"],
            "pretrained_source": "GastroNet-5M",
            "fine_tuned_on": "CD_endoscopy_images",
            "task": "abnormal_image_classification",
            "best_val_epoch": int(best_epoch),
            "final_train_epochs": int(best_epoch),
        }
        with open(hf_encoder_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(hf_config, f, indent=2)

        print(f"[Stage1] HuggingFace格式已保存: {hf_encoder_dir}")
    except ImportError:
        print("[Stage1][WARN] safetensors未安装，跳过HF格式保存")

    print("\n" + "=" * 80)
    print("[Stage1] 使用测试集进行最终评估...")
    print("=" * 80)

    model.to(device)
    test_metrics = evaluate(model, dl_te, device, use_amp=use_amp, channels_last=cfg.get("channels_last", True), gpu_normalize=gpu_normalize)
    full_train_metrics = evaluate(model, dl_full, device, use_amp=use_amp, channels_last=cfg.get("channels_last", True), gpu_normalize=gpu_normalize)

    best_val_metrics = history[best_epoch - 1] if best_epoch <= len(history) else history[-1]
    with open(stage1_dir / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": int(best_epoch),
            "best_val_metrics": best_val_metrics,
            "full_train_metrics": full_train_metrics,
            "test_metrics": test_metrics,
            "auc_gate_passed": bool(test_metrics["auc"] >= stage_cfg["auc_gate"]),
            "n_missing_images": int(len(missing_df)),
            "total_params_M": float(total_params / 1e6),
            "encoder_params_M": float(encoder_params / 1e6),
            "classifier_params_M": float(classifier_params / 1e6),
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Stage1 训练完成!")
    print("=" * 80)
    print(f"[Stage1] 最佳验证epoch = {best_epoch}")
    print(f"[Stage1] 全量训练后测试集评估:")
    print(f"[Stage1]   test_auc={test_metrics['auc']:.4f} test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['f1']:.4f}")
    print(f"[Stage1]   test_pos_rate={test_metrics['pos_rate']:.4f}")
    print(f"[Stage1] encoder_ckpt={encoder_ckpt_path}")

    if test_metrics["auc"] < stage_cfg["auc_gate"]:
        print(f"[Stage1][WARN] abnormal AUC={test_metrics['auc']:.4f} < {stage_cfg['auc_gate']:.2f}; 建议增加epochs或调整参数")

    run_comprehensive_evaluation(
        model, test_img, cfg["image_dir"], device, stage_cfg["image_size"],
        stage_cfg.get("abnormal_logit_bias", 0.0), out_dir
    )

if __name__ == "__main__":
    main()
