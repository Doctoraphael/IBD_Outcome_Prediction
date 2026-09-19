import argparse
import json
import math
import random
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from safetensors.torch import save_file as safetensors_save_file

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def save_hf_model(model: nn.Module, save_dir: Path, model_cfg: dict, model_name: str = "TransformerDualHead"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    hf_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder."):
            continue
        hf_state_dict[k] = v

    safetensors_path = save_dir / "model.safetensors"
    safetensors_save_file(hf_state_dict, str(safetensors_path))

    config = {
        "model_type": model_name,
        "architecture": "TransformerDualHead",
        "task": "dual_head_classification",
        "heads": ["remission", "healing"],
        "backbone": "dinov2_vit_base_patch14_reg4",
        "image_size": 336,
        "d_model": model.d_model,
        "instance_dropout": getattr(model, "instance_dropout", 0.0),
        **model_cfg
    }

    config_path = save_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    generation_config = {
        "model_type": model_name,
        "task": "classification",
        "num_labels": 2,
        "remission_threshold": 0.5,
        "healing_threshold": 0.5,
    }
    gen_config_path = save_dir / "generation_config.json"
    with open(gen_config_path, "w", encoding="utf-8") as f:
        json.dump(generation_config, f, ensure_ascii=False, indent=2)

    readme_content = f"""---
language:
- zh
- en
license: apache-2.0
model_name: {model_name}
tags:
- biomedical
- image-classification
- mil
- transformer
- ibd
task_categories:
- image-classification
- multi-label-classification
task_ids:
- biological-agent-outcome-prediction
---

# {model_name} - Transformer Dual-Head Model for IBD Treatment Outcome Prediction

## Model Description

This model predicts treatment outcomes for IBD (Inflammatory Bowel Disease) patients using biological agents:
- **remission**: Clinical remission prediction
- **healing**: Mucosal healing prediction

## Architecture

- **Backbone**: {config.get('backbone', 'dinov2_vit_base_patch14_reg4')}
- **Model Type**: Transformer Dual-Head
- **Input**: Medical images + clinical features (optional)

## Usage

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("{save_dir.name}")
model.eval()

with torch.no_grad():
    outputs = model(clinical, drug, bag_images, bag_mask)
```

## Training Configuration

- Learning rate: {model_cfg.get('lr', 'N/A')}
- Batch size: {model_cfg.get('batch_size_bag', 'N/A')}
- Epochs: {model_cfg.get('epochs', 'N/A')}
- Dropout: {model_cfg.get('dropout', 'N/A')}

## Performance

See stage3_model_summary.json for detailed metrics.
"""

    readme_path = save_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"  [HF] Model saved to {save_dir}")
    print(f"       - model.safetensors ({len(hf_state_dict)} tensors)")
    print(f"       - config.json")
    print(f"       - generation_config.json")
    print(f"       - README.md")

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
    if x.dim() == 5:
        mean = mean.view(1, 1, 3, 1, 1)
        std = std.view(1, 1, 3, 1, 1)
    return (x - mean) / std

def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

def find_optimal_threshold(y_true, y_prob, n_steps=20):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_probs = y_prob[pos_mask]
    neg_probs = y_prob[neg_mask]

    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.5, None

    best_thresh, best_separation = 0.5, -1
    best_f1_at_best = 0

    for t in np.arange(0.1, 0.9, 0.8 / n_steps):
        tpr = np.mean(pos_probs >= t)
        fpr = np.mean(neg_probs >= t)
        separation = tpr - fpr
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if separation > best_separation:
            best_separation = separation
            best_thresh = t
            best_f1_at_best = f1

    return best_thresh, best_f1_at_best

def compute_binary_metrics(y_true, y_prob, use_optimal_thresh=False):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if use_optimal_thresh:
        thresh, _ = find_optimal_threshold(y_true, y_prob)
    else:
        thresh = 0.5
    y_pred = (y_prob >= thresh).astype(int)
    return {
        "auc": safe_auc(y_true, y_prob),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "optimal_thresh": thresh if use_optimal_thresh else 0.5,
    }

def compute_binary_metrics_with_thresh(y_true, y_prob, thresh):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thresh).astype(int)
    return {
        "auc": safe_auc(y_true, y_prob),
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "thresh": thresh,
    }

def build_loader_kwargs(num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int | None):
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers and num_workers > 0),
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs

def stratified_sample_indices(labels, n_samples, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    valid_mask = np.array(labels) >= 0
    valid_indices = np.where(valid_mask)[0]
    valid_labels = np.array(labels)[valid_mask]

    if len(valid_indices) <= n_samples:
        return list(valid_indices)

    abnormal_idx = valid_indices[valid_labels == 1]
    normal_idx = valid_indices[valid_labels == 0]

    n_abnormal = len(abnormal_idx)
    n_normal = len(normal_idx)
    total = n_abnormal + n_normal

    n_sample_abnormal = int(round(n_samples * n_abnormal / total))
    n_sample_normal = n_samples - n_sample_abnormal

    n_sample_abnormal = min(n_sample_abnormal, n_abnormal)
    n_sample_normal = min(n_sample_normal, n_normal)

    sampled_abnormal = np.random.choice(abnormal_idx, n_sample_abnormal, replace=False) if n_sample_abnormal > 0 else np.array([], dtype=int)
    sampled_normal = np.random.choice(normal_idx, n_sample_normal, replace=False) if n_sample_normal > 0 else np.array([], dtype=int)

    sampled = np.concatenate([sampled_abnormal, sampled_normal])
    np.random.shuffle(sampled)

    return list(sampled)

def compute_clinical_stats(train_df: pd.DataFrame, clinical_cols: list):
    arr = train_df[clinical_cols].astype(np.float32).values
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


class ImageBagDataset(Dataset):
    def __init__(self, patient_df: pd.DataFrame, image_df: pd.DataFrame,
                 image_dir: str, transform,
                 clinical_cols: list,
                 drug_col: str = "drug_enc",
                 mode: str = "both",
                 bag_size: int = 48,
                 max_images_per_patient: int = 100,
                 training: bool = True,
                 clinical_mean=None, clinical_std=None):
        self.mode = mode
        self.max_images_per_patient = max_images_per_patient
        self.bag_size = bag_size
        self.training = training

        patient_df = patient_df.reset_index(drop=True).copy()
        patient_df["patient_id"] = patient_df["patient_id"].astype(str).map(normalize_patient_id)
        self.patient_ids = patient_df["patient_id"].astype(str).tolist()

        image_df = image_df.copy()
        image_df["patient_id"] = image_df["patient_id"].astype(str).map(normalize_patient_id)
        image_df["image"] = image_df["image"].astype(str).map(normalize_image_name)

        self.patient_images = {}
        for pid in self.patient_ids:
            pat_imgs = image_df[image_df["patient_id"] == pid]
            img_paths = []
            img_labels = []
            for _, row in pat_imgs.iterrows():
                path = resolve_image_path(image_dir, pid, row["image"])
                if path is not None:
                    img_paths.append(str(path))
                    img_labels.append(int(row["annotated_label"]))
            self.patient_images[pid] = {"paths": img_paths, "labels": img_labels}

        self.clinical_cols = clinical_cols
        self.drug_col = drug_col

        clin_np = patient_df[clinical_cols].astype(np.float32).values
        if clinical_mean is None or clinical_std is None:
            clinical_mean = clin_np.mean(axis=0, keepdims=True)
            clinical_std = clin_np.std(axis=0, keepdims=True)
        clinical_std = np.where(np.asarray(clinical_std) < 1e-6, 1.0, clinical_std)
        self.clinical_mean = np.asarray(clinical_mean, dtype=np.float32).reshape(1, -1)
        self.clinical_std = np.asarray(clinical_std, dtype=np.float32).reshape(1, -1)
        self.clinical_np = ((clin_np - self.clinical_mean) / self.clinical_std).astype(np.float32)

        self.drug_np = patient_df[drug_col].astype(np.float32).values
        self.remission = patient_df["remission"].astype(np.float32).values
        self.healing = patient_df["healing"].astype(np.float32).values
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def _pad_or_sample_bag(self, feats_or_paths, labels, training: bool):
        n = len(feats_or_paths)
        if n == 0:
            raise ValueError("Empty bag encountered.")

        if n <= self.bag_size:
            pad = self.bag_size - n
            if pad > 0:
                if isinstance(feats_or_paths[0], str):
                    feats_or_paths = feats_or_paths + ["__PAD__"] * pad
                else:
                    pad_tensors = [torch.zeros_like(feats_or_paths[0]) for _ in range(pad)]
                    feats_or_paths = feats_or_paths + pad_tensors
                labels = list(labels) + [-1] * pad
            mask = np.zeros((self.bag_size,), dtype=np.bool_)
            mask[n:] = True
            n_valid = n
        else:
            if training:
                idx = np.random.choice(n, self.bag_size, replace=False)
            else:
                idx = np.linspace(0, n - 1, self.bag_size, dtype=int)
            feats_or_paths = [feats_or_paths[i] for i in idx]
            labels = [labels[i] for i in idx]
            mask = np.zeros((self.bag_size,), dtype=np.bool_)
            n_valid = self.bag_size

        return feats_or_paths, labels, mask, n_valid

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        clinical = self.clinical_np[idx]
        drug = self.drug_np[idx]

        img_info = self.patient_images.get(pid, {"paths": [], "labels": []})
        img_paths = img_info["paths"]
        img_labels = img_info["labels"]

        if len(img_paths) > self.max_images_per_patient:
            if self.training:
                sampled_idx = stratified_sample_indices(
                    img_labels,
                    self.max_images_per_patient,
                    random_state=None
                )
            else:
                n = len(img_paths)
                sampled_idx = np.linspace(0, n - 1, self.max_images_per_patient, dtype=int).tolist()
            img_paths = [img_paths[i] for i in sampled_idx]
            img_labels = [img_labels[i] for i in sampled_idx]

        images = []
        valid_labels = []
        for path in img_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = self.transform(img)
                images.append(img)
                valid_labels.append(img_labels[len(images) - 1])
            except Exception:
                continue

        if len(images) == 0:
            img_size = self.transform.transforms[0].size[0] if hasattr(self.transform.transforms[0], 'size') else 336
            images = [torch.zeros(3, img_size, img_size)]
            valid_labels = [-1]

        images, labels, mask, n_valid = self._pad_or_sample_bag(images, valid_labels, self.training)
        bag_images = torch.stack(images) if isinstance(images[0], torch.Tensor) else None

        return {
            "patient_id": pid,
            "clinical": torch.tensor(clinical, dtype=torch.float32),
            "drug": torch.tensor(float(drug), dtype=torch.float32),
            "bag_images": bag_images,
            "bag_mask": torch.tensor(mask, dtype=torch.bool),
            "img_labels": torch.tensor(labels, dtype=torch.long),
            "n_valid": torch.tensor(n_valid, dtype=torch.long),
            "remission": torch.tensor(float(self.remission[idx]), dtype=torch.float32),
            "healing": torch.tensor(float(self.healing[idx]), dtype=torch.float32),
        }


class GastroNetEncoderWrapper(nn.Module):
    def __init__(self, backbone_name: str, image_size: int, ckpt_path: str, unfreeze_last_n_blocks: int = 0, pretrained_dinov2: bool = False):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if pretrained_dinov2:
            encoder_state = ckpt["teacher"]
            clean_state = {}
            for k, v in encoder_state.items():
                if k.startswith("backbone."):
                    k = k[len("backbone."):]
                if k.startswith("dino_head.") or k in ["register_tokens", "mask_token", "reg_token"]:
                    continue
                clean_state[k] = v
        else:
            encoder_state = ckpt["encoder_state_dict"]
            clean_state = {}
            for k, v in encoder_state.items():
                if k.startswith("encoder."):
                    k = k[len("encoder."):]
                clean_state[k] = v

        self.encoder = timm.create_model(backbone_name, pretrained=False, img_size=image_size, num_classes=0)

        model_state = self.encoder.state_dict()
        load_state = {}
        for k, v in clean_state.items():
            if k == "pos_embed":
                continue
            if k in model_state:
                load_state[k] = v
        self.encoder.load_state_dict(load_state, strict=False)

        for p in self.encoder.parameters():
            p.requires_grad = False
        self._n_unfrozen_blocks = unfreeze_last_n_blocks
        if unfreeze_last_n_blocks > 0:
            self._unfreeze_last_n_blocks(unfreeze_last_n_blocks)

    def _unfreeze_last_n_blocks(self, n_blocks: int):
        total_blocks = 12
        for name, param in self.encoder.named_parameters():
            if 'blocks.' in name:
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'blocks' and i + 1 < len(parts):
                        block_num = int(parts[i + 1])
                        if block_num >= total_blocks - n_blocks:
                            param.requires_grad = True
        for name, param in self.encoder.named_parameters():
            if 'norm' in name.lower():
                param.requires_grad = True

    def forward(self, x):
        return self.encoder.forward_features(x)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerDualHead(nn.Module):
    def __init__(self, encoder: nn.Module, n_clinical: int, n_drug: int = 1,
                 mode: str = "image_only", d_model: int = 384, n_heads: int = 4,
                 n_layers: int = 2, dim_ffn: int = 1536, dropout: float = 0.2,
                 instance_dropout: float = 0.15, modality_dropout: float = 0.15,
                 use_checkpoint: bool = True,
                 warmup_epochs: int = 0):
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        self.instance_dropout = instance_dropout
        self.modality_dropout = modality_dropout
        self.use_checkpoint = use_checkpoint
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        feature_dim = 768

        self.encoder = encoder
        self.img_proj = nn.Linear(feature_dim, d_model)

        if mode == "both":
            self.clinical_encoder = nn.Sequential(
                nn.Linear(n_clinical, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.clinical_scale = 0.1

        self.drug_encoder = nn.Sequential(
            nn.Linear(n_drug, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model),
        )

        self.cross_attn_drug_to_patient = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_drug_norm = nn.LayerNorm(d_model)

        self.bias_net_rem = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        self.bias_net_hea = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head_rem = nn.Linear(d_model, 1)
        self.head_hea = nn.Linear(d_model, 1)

        torch.manual_seed(42)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.xavier_uniform_(self.head_rem.weight)
            nn.init.constant_(self.head_rem.bias, 0.0)
            nn.init.xavier_uniform_(self.head_hea.weight)
            nn.init.constant_(self.head_hea.bias, 0.0)
            for net in [self.bias_net_rem, self.bias_net_hea]:
                for m in net.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.constant_(m.bias, 0.0)

    def extract_image_features(self, images, chunk_size: int = 20):
        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        total = B * N
        feats_list = []

        for i in range(0, total, chunk_size):
            chunk = images_flat[i:i + chunk_size]
            if self.use_checkpoint and self.training:
                for sub_start in range(0, chunk.size(0), 8):
                    sub_chunk = chunk[sub_start:sub_start+8]
                    sub_feats = torch.utils.checkpoint.checkpoint(
                        self.encoder, sub_chunk, use_reentrant=False
                    )
                    if sub_start == 0:
                        feats_chunk = sub_feats[:, 0, :]
                    else:
                        feats_chunk = torch.cat([feats_chunk, sub_feats[:, 0, :]], dim=0)
            else:
                feats_chunk = self.encoder(chunk)[:, 0, :]
            feats_list.append(feats_chunk)

        feats_all = torch.cat(feats_list, dim=0)
        return feats_all.view(B, N, -1)

    def forward(self, clinical, drug, bag_images, bag_mask, task=None):
        B, N = bag_images.shape[:2]

        if self.training and self.instance_dropout > 0:
            inst_mask = (torch.rand(B, N, device=bag_images.device) > self.instance_dropout).float()
            inst_mask = inst_mask.unsqueeze(-1)
        else:
            inst_mask = torch.ones(B, N, 1, device=bag_images.device)

        img_feats = self.extract_image_features(bag_images)
        img_feats = self.img_proj(img_feats) * inst_mask

        drug_repr = self.drug_encoder(drug.unsqueeze(1)).squeeze(1)

        if self.mode == "both":
            clin_repr = self.clinical_encoder(clinical)
            clin_token = clin_repr.unsqueeze(1) * self.clinical_scale

            if self.modality_dropout > 0 and self.training:
                mask = torch.bernoulli(torch.full((B, 1, 1), 1 - self.modality_dropout, device=clin_token.device, dtype=clin_token.dtype))
                clin_token = clin_token * mask

            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, img_feats, clin_token], dim=1)
            tokens = self.pos_encoder(tokens)

            output = self.transformer(tokens)

            patient_repr = output[:, 0, :]

            drug_reshaped = drug_repr.unsqueeze(1)
            patient_reshaped = patient_repr.unsqueeze(1)
            drug_attended, _ = self.cross_attn_drug_to_patient(
                drug_reshaped, patient_reshaped, patient_reshaped
            )
            drug_attended = drug_attended.squeeze(1)
            drug_attended = self.cross_attn_drug_norm(drug_repr + drug_attended)

            bias_rem = self.bias_net_rem(drug_attended)
            bias_hea = self.bias_net_hea(drug_attended)

            rem_hidden = patient_repr + bias_rem
            hea_hidden = patient_repr + bias_hea

            rem_logits = self.head_rem(rem_hidden).squeeze(-1)
            hea_logits = self.head_hea(hea_hidden).squeeze(-1)

            return {
                "remission_logits": rem_logits,
                "healing_logits": hea_logits,
                "remission_prob": torch.sigmoid(rem_logits),
                "healing_prob": torch.sigmoid(hea_logits),
                "bag_repr": patient_repr,
                "instance_features": img_feats,
            }
        else:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, img_feats], dim=1)
            tokens = self.pos_encoder(tokens)
            output = self.transformer(tokens)
            patient_repr = output[:, 0, :]

        drug_reshaped = drug_repr.unsqueeze(1)
        patient_reshaped = patient_repr.unsqueeze(1)
        drug_attended, _ = self.cross_attn_drug_to_patient(
            drug_reshaped, patient_reshaped, patient_reshaped
        )
        drug_attended = drug_attended.squeeze(1)
        drug_attended = self.cross_attn_drug_norm(drug_repr + drug_attended)

        bias_rem = self.bias_net_rem(drug_attended)
        bias_hea = self.bias_net_hea(drug_attended)

        rem_hidden = patient_repr + bias_rem
        hea_hidden = patient_repr + bias_hea

        rem_logits = self.head_rem(rem_hidden).squeeze(-1)
        hea_logits = self.head_hea(hea_hidden).squeeze(-1)

        return {
            "remission_logits": rem_logits,
            "healing_logits": hea_logits,
            "remission_prob": torch.sigmoid(rem_logits),
            "healing_prob": torch.sigmoid(hea_logits),
            "bag_repr": patient_repr,
            "instance_features": img_feats,
        }

def build_image_model(model_name: str, encoder: nn.Module, n_clinical: int, n_drug: int, cfg_stage23: dict):
    d_model = cfg_stage23.get("d_model", 128)
    n_heads = cfg_stage23.get("n_heads", 2)
    n_layers = cfg_stage23.get("n_layers", 1)
    dim_ffn = cfg_stage23.get("dim_ffn", 256)
    dropout = cfg_stage23.get("dropout", 0.55)
    instance_dropout = cfg_stage23.get("instance_dropout", 0.4)
    modality_dropout = cfg_stage23.get("modality_dropout", 0.15)
    use_checkpoint = cfg_stage23.get("use_checkpoint", True)
    warmup_epochs = cfg_stage23.get("warmup_epochs", 0)

    return TransformerDualHead(
        encoder, n_clinical, n_drug=n_drug,
        mode="image_only" if "image_only" in model_name else "both",
        d_model=d_model, n_heads=n_heads, n_layers=n_layers, dim_ffn=dim_ffn,
        dropout=dropout, instance_dropout=instance_dropout, modality_dropout=modality_dropout,
        use_checkpoint=use_checkpoint,
        warmup_epochs=warmup_epochs
    )


def dualhead_loss_with_constraint(outputs, rem_labels, hea_labels, crit,
                                  constraint_weight: float = 0.1,
                                  label_smoothing: float = 0.02):
    if label_smoothing > 0:
        rem_labels_smooth = rem_labels * (1 - label_smoothing) + label_smoothing / 2
        hea_labels_smooth = hea_labels * (1 - label_smoothing) + label_smoothing / 2
    else:
        rem_labels_smooth = rem_labels
        hea_labels_smooth = hea_labels

    rem_logits = outputs["remission_logits"]
    hea_logits = outputs["healing_logits"]
    rem_prob = torch.sigmoid(rem_logits)
    hea_prob = torch.sigmoid(hea_logits)

    rem_bce = crit(rem_logits, rem_labels_smooth)

    rem_mask = rem_labels_smooth >= 0.5
    if rem_mask.sum() > 0:
        hea_bce = crit(hea_logits[rem_mask], hea_labels_smooth[rem_mask])
    else:
        hea_bce = torch.tensor(0.0, device=rem_logits.device)


    constraint_A = (1 - rem_labels) * hea_prob

    constraint_B = hea_labels * torch.clamp(1 - rem_prob, min=0)

    constraint_loss = (constraint_A + constraint_B).mean()

    total = rem_bce + hea_bce + constraint_weight * constraint_loss

    return total, rem_bce.detach(), hea_bce.detach(), constraint_loss.detach()

def compute_class_labels(rem_labels, hea_labels):
    class_labels = np.zeros(len(rem_labels), dtype=int)
    for i in range(len(rem_labels)):
        if rem_labels[i] == 0:
            class_labels[i] = 0
        elif hea_labels[i] == 0:
            class_labels[i] = 1
        else:
            class_labels[i] = 2
    return class_labels.tolist()

def run_eval_dualhead(model, loader, device, use_amp: bool = True, debug: bool = False):
    model.eval()
    out = {
        "remission": {"y": [], "p": []},
        "healing": {"y": [], "p": []},
        "class_0": {"y": [], "p": []},
        "class_1": {"y": [], "p": []},
        "class_2": {"y": [], "p": []},
    }
    debug_info = []
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device, non_blocking=True)
            drug = batch["drug"].to(device, non_blocking=True)
            if batch["bag_images"] is not None:
                bag_images = batch["bag_images"].to(device, non_blocking=True)
                bag_mask = batch["bag_mask"].to(device, non_blocking=True)
            else:
                bag_images = None
                bag_mask = None

            rem_labels = batch["remission"]
            hea_labels = batch["healing"]

            with autocast_ctx(use_amp):
                outputs = model(clinical, drug, bag_images, bag_mask)

            rem_prob = outputs["remission_prob"].cpu().numpy().tolist()
            hea_prob = outputs["healing_prob"].cpu().numpy().tolist()

            if debug and len(debug_info) == 0:
                debug_info.append({
                    "drug": drug.cpu().numpy()[:3].tolist(),
                    "rem_labels": rem_labels.numpy()[:3].tolist(),
                    "hea_labels": hea_labels.numpy()[:3].tolist(),
                    "rem_prob": rem_prob[:3],
                    "hea_prob": hea_prob[:3],
                })

            out["remission"]["y"].extend(rem_labels.numpy().tolist())
            out["remission"]["p"].extend(rem_prob)
            out["healing"]["y"].extend(hea_labels.numpy().tolist())
            out["healing"]["p"].extend(hea_prob)

            rem_prob_arr = np.array(rem_prob)
            hea_prob_arr = np.array(hea_prob)
            class_0_prob = 1 - rem_prob_arr
            class_1_prob = rem_prob_arr * (1 - hea_prob_arr)
            class_2_prob = rem_prob_arr * hea_prob_arr
            class_probs = np.stack([class_0_prob, class_1_prob, class_2_prob], axis=1)

            class_labels = compute_class_labels(rem_labels.numpy(), hea_labels.numpy())
            for c in range(3):
                mask = np.array(class_labels) == c
                out[f"class_{c}"]["y"].extend(class_labels)
                out[f"class_{c}"]["p"].extend(class_probs[:, c].tolist())

    rem_metrics = compute_binary_metrics(out["remission"]["y"], out["remission"]["p"])
    hea_metrics = compute_binary_metrics(out["healing"]["y"], out["healing"]["p"])

    class_metrics = {}
    for c in range(3):
        y_true = (np.array(out[f"class_{c}"]["y"]) == c).astype(int)
        y_prob = out[f"class_{c}"]["p"]
        class_metrics[f"class_{c}"] = {
            "auc": safe_auc(y_true, y_prob),
            "acc": accuracy_score(y_true, (np.array(y_prob) > 0.5).astype(int)),
        }

    if debug:
        print(f"[DEBUG] First batch: drug={debug_info[0]['drug']}, rem_labels={debug_info[0]['rem_labels']}, "
              f"hea_labels={debug_info[0]['hea_labels']}, rem_prob={debug_info[0]['rem_prob']}, "
              f"hea_prob={debug_info[0]['hea_prob']}")

    return {
        "remission": rem_metrics,
        "healing": hea_metrics,
        "class_metrics": class_metrics,
        "raw": out,
    }

def run_eval_dualhead_with_thresh(model, loader, device, thresh, use_amp=True):
    model.eval()
    out = {
        "remission": {"y": [], "p": []},
        "healing": {"y": [], "p": []},
    }
    with torch.no_grad():
        for batch in loader:
            clinical = batch["clinical"].to(device, non_blocking=True)
            drug = batch["drug"].to(device, non_blocking=True)
            if batch["bag_images"] is not None:
                bag_images = batch["bag_images"].to(device, non_blocking=True)
                bag_mask = batch["bag_mask"].to(device, non_blocking=True)
            else:
                bag_images = None
                bag_mask = None

            rem_labels = batch["remission"]
            hea_labels = batch["healing"]

            with autocast_ctx(use_amp):
                outputs = model(clinical, drug, bag_images, bag_mask)

            rem_prob = outputs["remission_prob"].cpu().numpy().tolist()
            hea_prob = outputs["healing_prob"].cpu().numpy().tolist()

            out["remission"]["y"].extend(rem_labels.numpy().tolist())
            out["remission"]["p"].extend(rem_prob)
            out["healing"]["y"].extend(hea_labels.numpy().tolist())
            out["healing"]["p"].extend(hea_prob)

    rem_metrics = compute_binary_metrics_with_thresh(out["remission"]["y"], out["remission"]["p"], thresh)
    hea_metrics = compute_binary_metrics_with_thresh(out["healing"]["y"], out["healing"]["p"], thresh)

    return {
        "remission": rem_metrics,
        "healing": hea_metrics,
        "raw": out,
    }

def train_and_evaluate_on_test(model_name, train_df, test_df, image_df, cfg, encoder, clinical_cols, n_drug, device, use_amp):
    stage_cfg = cfg["stage23"]

    if "image_only" in model_name:
        mode = "image_only"
    else:
        mode = "both"

    model_cfg = {**stage_cfg, "model_name": model_name}
    bag_size = stage_cfg.get("bag_size", 100)
    max_images = stage_cfg.get("max_images_per_patient", 100)
    max_epochs = stage_cfg.get("cv_max_epochs", 100)
    patience = stage_cfg.get("cv_early_stop_patience", 20)
    min_delta = 0.005
    warmup_epochs = stage_cfg.get("warmup_epochs", 5)
    warmup_encoder_lr = stage_cfg.get("warmup_encoder_lr", 5e-7)
    warmup_head_lr = stage_cfg.get("warmup_head_lr", 1e-6)
    clinical_low_lr_epochs = stage_cfg.get("clinical_low_lr_epochs", 5)

    image_size = cfg["stage1"]["image_size"]
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.06, 0.06, 0.04, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    clinical_mean, clinical_std = compute_clinical_stats(train_df, clinical_cols)

    train_ds = ImageBagDataset(
        train_df, image_df, cfg["image_dir"], train_tf, clinical_cols,
        mode=mode, bag_size=bag_size, max_images_per_patient=max_images,
        training=True, clinical_mean=clinical_mean, clinical_std=clinical_std
    )
    test_ds = ImageBagDataset(
        test_df, image_df, cfg["image_dir"], eval_tf, clinical_cols,
        mode=mode, bag_size=bag_size, max_images_per_patient=max_images,
        training=False, clinical_mean=clinical_mean, clinical_std=clinical_std
    )

    nw = stage_cfg.get("num_workers_ima", 0)
    train_loader = DataLoader(train_ds, batch_size=stage_cfg.get("batch_size_bag", 4),
                             shuffle=True, num_workers=nw, pin_memory=True, persistent_workers=(nw>0))
    test_loader = DataLoader(test_ds, batch_size=stage_cfg.get("batch_size_bag", 4),
                           shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=False)

    model = build_image_model(model_name, encoder, len(clinical_cols), n_drug, model_cfg)
    model.to(device)

    head_lr = float(stage_cfg.get("lr", 1e-5))
    encoder_lr = float(stage_cfg.get("encoder_lr", 5e-6))
    clinical_encoder_lr = float(stage_cfg.get("clinical_encoder_lr", 1e-6))
    weight_decay = float(stage_cfg.get("weight_decay", 0.05))

    param_groups = []
    head_params = []
    encoder_params = []
    clinical_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder" in n:
            encoder_params.append(p)
        elif "clinical_encoder" in n:
            clinical_params.append(p)
        elif "cross_attn" in n:
            head_params.append(p)
        else:
            head_params.append(p)

    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "name": "heads"})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr, "name": "encoder"})
    if clinical_params:
        param_groups.append({"params": clinical_params, "lr": clinical_encoder_lr, "name": "clinical"})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_pos = (train_df["remission"] == 1).sum()
    n_neg = (train_df["remission"] == 0).sum()
    pos_weight = 1.5
    print(f"  Using pos_weight for remission: {pos_weight:.2f} (CD 75% positive, avoid being 'drowned' by majority class)")
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_epoch = 0
    best_rem_auc = 0.0
    best_hea_auc = 0.0
    best_rem_acc = 0.0
    best_rem_f1 = 0.0
    best_test_metrics = None
    epochs_no_improve = 0

    print(f"  Training on Train, evaluating on Test each epoch (max_epochs={max_epochs}, patience={patience}, warmup={warmup_epochs})...")

    base_lrs = {pg["name"]: pg["lr"] for pg in param_groups}
    clinical_low_lr = clinical_encoder_lr * 0.1

    for epoch in range(max_epochs):
        model.train()
        model.current_epoch = epoch

        if epoch < warmup_epochs:
            phase = "WARMUP"
            for pg in optimizer.param_groups:
                if pg["name"] == "encoder":
                    pg["lr"] = warmup_encoder_lr
                elif pg["name"] == "heads":
                    pg["lr"] = warmup_head_lr
                elif pg["name"] == "clinical":
                    pg["lr"] = clinical_low_lr
        else:
            phase = "TRAINING"
            lr_ramp_epochs = 10
            progress = min((epoch - warmup_epochs) / lr_ramp_epochs, 1.0)
            for pg in optimizer.param_groups:
                if pg["name"] in base_lrs:
                    if pg["name"] == "encoder":
                        pg["lr"] = warmup_encoder_lr + progress * (base_lrs[pg["name"]] - warmup_encoder_lr)
                    elif pg["name"] == "heads":
                        pg["lr"] = warmup_head_lr + progress * (base_lrs[pg["name"]] - warmup_head_lr)
                    elif pg["name"] == "clinical":
                        pg["lr"] = clinical_low_lr + progress * (base_lrs[pg["name"]] - clinical_low_lr)
                    else:
                        pg["lr"] = base_lrs[pg["name"]]

        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            clinical = batch["clinical"].to(device, non_blocking=True)
            drug = batch["drug"].to(device, non_blocking=True)
            bag_images = batch["bag_images"].to(device, non_blocking=True)
            bag_mask = batch["bag_mask"].to(device, non_blocking=True)
            rem_labels = batch["remission"].float().to(device, non_blocking=True)
            hea_labels = batch["healing"].float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(clinical, drug, bag_images, bag_mask)
                loss, _, _, _ = dualhead_loss_with_constraint(
                    outputs, rem_labels, hea_labels, crit,
                    constraint_weight=0.5,
                    label_smoothing=stage_cfg.get("label_smoothing", 0.02)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            n_batches += 1

        model.eval()
        test_metrics = run_eval_dualhead(model, test_loader, device, use_amp)
        rem_auc = test_metrics["remission"]["auc"]
        rem_acc = test_metrics["remission"]["acc"]
        rem_f1 = test_metrics["remission"]["f1"]
        hea_auc = test_metrics["healing"]["auc"]
        hea_acc = test_metrics["healing"]["acc"]
        hea_f1 = test_metrics["healing"]["f1"]

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  [{phase}] [Epoch {epoch+1}/{max_epochs}] loss={avg_loss:.4f} | rem: auc={rem_auc:.4f} acc={rem_acc:.4f} f1={rem_f1:.4f} | hea: auc={hea_auc:.4f} acc={hea_acc:.4f} f1={hea_f1:.4f}")

        if epoch >= warmup_epochs:
            if rem_auc > best_rem_auc + min_delta:
                best_rem_auc = rem_auc
                best_hea_auc = hea_auc
                best_rem_acc = rem_acc
                best_rem_f1 = rem_f1
                best_epoch = epoch + 1
                best_test_metrics = test_metrics
                epochs_no_improve = 0
                print(f"    -> New best! rem_auc={rem_auc:.4f}")

                stage3_dir = Path(cfg["output_dir"]) / "stage3"
                stage3_dir.mkdir(parents=True, exist_ok=True)
                model_ckpt_path = stage3_dir / f"{model_name}_best_model.pt"
                torch.save({
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "rem_auc": best_rem_auc,
                    "hea_auc": best_hea_auc,
                    "acc": best_rem_acc,
                    "f1": best_rem_f1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg,
                }, model_ckpt_path)

                hf_path = stage3_dir / f"{model_name}_best_model_hf"
                save_hf_model(model, hf_path, model_cfg, model_name=model_name)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

    print(f"  Best epoch: {best_epoch}, Best Test rem_auc: {best_rem_auc:.4f} (acc={best_rem_acc:.4f}, f1={best_rem_f1:.4f})")

    print("  Finding optimal threshold on training set...")
    model = build_image_model(model_name, encoder, len(clinical_cols), n_drug, model_cfg)
    model.to(device)
    ckpt = torch.load(stage3_dir / f"{model_name}_best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_eval_loader = DataLoader(train_ds, batch_size=stage_cfg.get("batch_size_bag", 4),
                                  shuffle=False, num_workers=nw, pin_memory=True, persistent_workers=False)
    train_raw = run_eval_dualhead(model, train_eval_loader, device, use_amp)
    train_probs = train_raw["raw"]["remission"]["p"]
    train_labels = train_raw["raw"]["remission"]["y"]

    opt_thresh, opt_f1 = find_optimal_threshold(train_labels, train_probs)
    print(f"  Optimal threshold: {opt_thresh:.4f} (train F1={opt_f1:.4f})")
    test_opt_metrics = run_eval_dualhead_with_thresh(model, test_loader, device, opt_thresh, use_amp)
    print(f"  Test with optimal threshold ({opt_thresh:.4f}): rem_acc={test_opt_metrics['remission']['acc']:.4f}, rem_f1={test_opt_metrics['remission']['f1']:.4f}")
    if best_test_metrics is not None:
        best_test_metrics["remission"]["acc"] = test_opt_metrics["remission"]["acc"]
        best_test_metrics["remission"]["f1"] = test_opt_metrics["remission"]["f1"]
        best_test_metrics["remission"]["optimal_thresh"] = opt_thresh

    del model, optimizer, scaler, train_loader, test_loader
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "best_epoch": best_epoch,
        "best_rem_auc": best_rem_auc,
        "best_hea_auc": best_hea_auc,
        "test_metrics": best_test_metrics,
    }

def bootstrap_confidence_interval(y_true, y_prob, n_bootstrap=2000, seed=42):
    np.random.seed(seed)
    n = len(y_true)
    aucs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        y_t = np.array(y_true)[indices]
        y_p = np.array(y_prob)[indices]

        if len(np.unique(y_t)) < 2:
            continue

        try:
            auc = roc_auc_score(y_t, y_p)
            aucs.append(auc)
        except ValueError:
            continue

    aucs = np.array(aucs)
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_median": float(np.median(aucs)),
        "ci_lower": float(np.percentile(aucs, 2.5)),
        "ci_upper": float(np.percentile(aucs, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage3: Transformer Dual-Head Models - Dual Task Prediction")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))
    seed_everything(cfg["seed"])
    device = configure_gpu(cfg)
    use_amp = amp_enabled(cfg, device)

    print("=" * 80)
    print("Stage 3: Transformer Dual-Head Models - Dual Task (remission + healing)")
    print("=" * 80)
    print(f"[Stage3] device={device} | cuda_available={torch.cuda.is_available()} | use_amp={use_amp}")
    if device.type == "cuda":
        try:
            print(f"[Stage3] gpu={torch.cuda.get_device_name(0)}")
            print(f"[Stage3] gpu_memory={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception:
            pass

    pat_df = pd.read_csv(cfg["pat_csv"]).copy()
    pat_df = pat_df.dropna(subset=["healing"]).reset_index(drop=True)
    pat_df["patient_id"] = pat_df["patient_id"].astype(str).map(normalize_patient_id)
    pat_df["drug_enc"] = (pat_df["drug"].astype(str) == "UST").astype(np.float32)

    stage_cfg = cfg["stage23"]
    clinical_cols = list(stage_cfg["clinical_cols"])

    med = pat_df[clinical_cols].median(numeric_only=True)
    pat_df[clinical_cols] = pat_df[clinical_cols].fillna(med)

    img_df = pd.read_csv(cfg["img_csv"]).copy()
    img_df = img_df[img_df["annotated_label"].isin([0, 1])].copy().reset_index(drop=True)

    cache_dir = Path(cfg["output_dir"]) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_drug = 1
    summary = {}

    image_models = ["image_only", "both"]

    print("\n数据集分层情况 (drug × healing):")
    print("=" * 60)
    for (drug, label), group in pat_df.groupby(["drug_enc", "healing"]):
        print(f"  (drug={int(drug)}, hea={int(label)}): n={len(group)}")

    encoder_ckpt = Path(cfg["gastronet_ckpt"])
    if not encoder_ckpt.exists():
        raise FileNotFoundError(f"预训练DINOv2模型不存在: {encoder_ckpt}")

    backbone_name = "vit_base_patch14_reg4_dinov2"
    image_size = 336
    unfreeze_blocks = cfg.get("stage23", {}).get("unfreeze_encoder_blocks", 0)
    if unfreeze_blocks > 0:
        print(f"\n[Stage3] Loading pretrained DINOv2 from: {encoder_ckpt}")
        print(f"[Stage3] Encoder: {backbone_name}, last {unfreeze_blocks} block(s) UNFROZEN")
    else:
        print(f"\n[Stage3] Loading pretrained DINOv2 from: {encoder_ckpt}")
        print(f"[Stage3] Encoder: {backbone_name}, fully FROZEN (no fine-tuning)")
    encoder = GastroNetEncoderWrapper(
        backbone_name,
        image_size,
        str(encoder_ckpt),
        unfreeze_last_n_blocks=unfreeze_blocks,
        pretrained_dinov2=True
    )
    encoder.to(device)
    encoder.eval()

    for model_name in image_models:
        print(f"\n{'='*80}")
        print(f"= Model: {model_name} (DualHead - remission + healing)")
        print(f"{'='*80}")

        split_info_path = cache_dir / "split_info_healing.json"
        if not split_info_path.exists():
            raise FileNotFoundError(f"统一划分文件不存在: {split_info_path}，请先运行Runner创建划分")

        split_info = json.load(open(split_info_path, "r", encoding="utf-8"))
        train_ids = set(split_info["train_patient_ids"])
        test_ids = set(split_info["test_patient_ids"])
        train_mask = pat_df["patient_id"].isin(train_ids)
        test_mask = pat_df["patient_id"].isin(test_ids)
        tr_all = pat_df[train_mask].reset_index(drop=True)
        te = pat_df[test_mask].reset_index(drop=True)

        print(f"\n[{model_name}] Train patients: {len(tr_all)}, Test patients: {len(te)}")

        print(f"\n[{model_name}] 划分后remission分层:")
        print("  Training set:")
        for (drug, label), group in tr_all.groupby(["drug_enc", "remission"]):
            print(f"    (drug={int(drug)}, rem={int(label)}): n={len(group)}")
        print("  Test set:")
        for (drug, label), group in te.groupby(["drug_enc", "remission"]):
            print(f"    (drug={int(drug)}, rem={int(label)}): n={len(group)}")

        print(f"\n[{model_name}] Training with evaluation on Test each epoch...")
        result = train_and_evaluate_on_test(
            model_name, tr_all, te, img_df, cfg, encoder, clinical_cols, n_drug, device, use_amp
        )
        best_epoch = result["best_epoch"]
        best_test_rem_auc = result["best_rem_auc"]
        best_test_hea_auc = result["best_hea_auc"]
        print(f"\n[{model_name}] Best epoch: {best_epoch}, Best Test rem AUC: {best_test_rem_auc:.4f}")

        print(f"\n[{model_name}] Computing Bootstrap CI (n=2000)...")
        rem_bootstrap = bootstrap_confidence_interval(
            result["test_metrics"]["raw"]["remission"]["y"],
            result["test_metrics"]["raw"]["remission"]["p"],
            n_bootstrap=2000, seed=cfg["seed"]
        )
        hea_bootstrap = bootstrap_confidence_interval(
            result["test_metrics"]["raw"]["healing"]["y"],
            result["test_metrics"]["raw"]["healing"]["p"],
            n_bootstrap=2000, seed=cfg["seed"]
        )

        result["epoch_selection"] = {
            "best_epoch": best_epoch,
            "best_test_rem_auc": best_test_rem_auc,
            "best_test_hea_auc": best_test_hea_auc,
        }
        result["bootstrap_ci"] = {
            "remission": rem_bootstrap,
            "healing": hea_bootstrap,
        }
        summary[model_name] = result

        gc.collect()
        torch.cuda.empty_cache()

    stage3_dir = Path(cfg["output_dir"]) / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)
    out_path = stage3_dir / "stage3_model_summary.json"
    json.dump(summary, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Stage3 全部图像模型训练完成 (双Head架构 - 主任务: remission, 辅助: healing)!")
    print("=" * 80)

    print("\n模型性能汇总 (主任务: remission, 辅助: healing):")
    print("=" * 100)
    print(f"{'Model':20s} | {'Epoch':>5s} | {'Test Rem AUC':>12s} | {'Test Hea (aux)':>14s} | {'Rem 95% CI':>15s} | {'Hea 95% CI':>15s}")
    print("-" * 100)
    for name, res in summary.items():
        test = res["test_metrics"]
        epoch_info = res.get("epoch_selection", {})
        boot = res.get("bootstrap_ci", {})
        rem_ci = boot.get("remission", {}).get("ci_lower", 0), boot.get("remission", {}).get("ci_upper", 0)
        hea_ci = boot.get("healing", {}).get("ci_lower", 0), boot.get("healing", {}).get("ci_upper", 0)
        print(f"{name:20s} | {epoch_info.get('best_epoch', 0):>5d} | {epoch_info.get('best_test_rem_auc', 0):>12.4f} | {test['remission']['auc']:>10.4f} | {test['healing']['auc']:>14.4f} | [{rem_ci[0]:.3f}, {rem_ci[1]:.3f}] | [{hea_ci[0]:.3f}, {hea_ci[1]:.3f}]")

    print("\n" + "=" * 80)
    print("详细信息（External Test）:")
    print("-" * 80)
    for name, res in summary.items():
        test = res["test_metrics"]
        boot = res.get("bootstrap_ci", {})
        print(f"\n{name}:")
        print(f"  Remission (primary): AUC={test['remission']['auc']:.4f}, Acc={test['remission']['acc']:.4f}, F1={test['remission']['f1']:.4f}")
        print(f"  Healing (auxiliary): AUC={test['healing']['auc']:.4f}, Acc={test['healing']['acc']:.4f}, F1={test['healing']['f1']:.4f}")
        print(f"  Bootstrap CI (Remission): [{boot.get('remission', {}).get('ci_lower', 0):.4f}, {boot.get('remission', {}).get('ci_upper', 0):.4f}]")
        print(f"  Bootstrap CI (Healing): [{boot.get('healing', {}).get('ci_lower', 0):.4f}, {boot.get('healing', {}).get('ci_upper', 0):.4f}]")

if __name__ == "__main__":
    main()
