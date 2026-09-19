# IBD Treatment Outcome Prediction

Author: Raphael Hong

Code for predicting clinical remission and mucosal healing outcomes in inflammatory bowel disease (IBD) patients treated with biological agents.

This repository contains the training and evaluation pipelines for two cohorts:

- **CD**: Crohn's disease
- **UC**: Ulcerative colitis

## Pipeline Overview

Each cohort follows a four-stage workflow:

| Stage | Script | Purpose |
|-------|--------|---------|
| 1 | `stage1_encoder.py` | Fine-tune a pretrained vision encoder for abnormal endoscopic image classification |
| 2 | `stage2_clinical.py` | Train clinical machine-learning baselines (Logistic Regression, Random Forest, SVM, XGBoost, MLP) |
| 3 | `stage3_transformer.py` | Train multimodal Transformer dual-head models (`image_only`, `both`) |
| 4 | `stage4_evaluate.py` | Evaluate all models, bootstrap CIs, ROC comparison, attention visualization |
| Runner | `run_pipeline.py` | Orchestrate stages 1–4 and generate the pipeline config |

### Model inputs

- Endoscopic images (multiple images per patient, treated as a bag)
- Clinical laboratory features
- Drug type as a conditioning variable

### Prediction targets

- `remission`: clinical remission
- `healing`: mucosal healing (constraint: healing = 1 implies remission = 1)

## Repository Layout

```
IBD_Outcome_Prediction/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── cd_scripts/          # Crohn's disease pipeline
│   ├── stage1_encoder.py
│   ├── stage2_clinical.py
│   ├── stage3_transformer.py
│   ├── stage4_evaluate.py
│   └── run_pipeline.py
└── uc_scripts/          # Ulcerative colitis pipeline
    ├── stage1_encoder.py
    ├── stage2_clinical.py
    ├── stage3_transformer.py
    ├── stage4_evaluate.py
    └── run_pipeline.py
```

## Requirements

See `requirements.txt`. Main dependencies:

- Python >= 3.10
- PyTorch
- timm
- scikit-learn
- XGBoost
- pandas / numpy
- matplotlib / seaborn
- safetensors
- joblib

Install:

```bash
pip install -r requirements.txt
```

## Data Layout

Place data under a cohort root directory (example for CD):

```
<root>/
├── imageCD/                          # images organized by patient_id
├── cd_image_analysis.csv             # image-level labels
├── cd_patient_summary.csv            # patient-level clinical labels
└── hf_cache/
    └── dinov2.pth                    # pretrained vision encoder checkpoint
```

For UC, use `imageUC/`, `uc_image_analysis.csv`, `uc_patient_summary.csv`.

### Expected CSV columns

**Patient CSV**

- `patient_id`
- `drug`
- `remission`
- `healing`
- clinical feature columns configured in the runner (`clinical_cols`)

**Image CSV**

- `patient_id`
- `image`
- `annotated_label` (0 = normal, 1 = abnormal)

### Drug encoding

- CD: `drug_enc = 1` if drug is UST, otherwise 0 (Anti-TNF)
- UC: `drug_enc = 1` if drug is VDZ, otherwise 0

## Running

### Full pipeline

```bash
# Crohn's disease
python cd_scripts/run_pipeline.py --root /path/to/cd_data --output-dir /path/to/cd_data/outputs_cd

# Ulcerative colitis
python uc_scripts/run_pipeline.py --root /path/to/uc_data --output-dir /path/to/uc_data/outputs_uc
```

Optional flags:

- `--config-out`: custom config path
- `--skip-stage1` / `--skip-stage2` / `--skip-stage3` / `--skip-stage4`

The runner writes a JSON config and creates a shared train/test split before launching stages.

### Individual stages

```bash
python cd_scripts/stage1_encoder.py --config /path/to/outputs_cd/cd_pipeline_config.json
python cd_scripts/stage2_clinical.py --config /path/to/outputs_cd/cd_pipeline_config.json
python cd_scripts/stage3_transformer.py --config /path/to/outputs_cd/cd_pipeline_config.json
python cd_scripts/stage4_evaluate.py --config /path/to/outputs_cd/cd_pipeline_config.json
```

## Pretrained Encoder

Stage 1 and Stage 3 require a pretrained DINOv2 / GastroNet-style vision checkpoint. Point `gastronet_ckpt` in the runner config to your local checkpoint path (e.g. `dinov2.pth`).

Default backbone name: `vit_base_patch14_reg4_dinov2`.

## Outputs

After a full run, the output directory typically contains:

```
outputs_*/
├── cache/                          # train/test split and intermediate logs
├── stage1/                         # encoder checkpoint and evaluation plots
├── stage2/                         # clinical model artifacts + hf_model/
├── stage3/                         # Transformer dual-head checkpoints
└── stage4/                         # metrics JSON, ROC curves, performance tables
```

## Notes for Reviewers

- Random seeds are fixed in the pipeline config (`seed`, `train_test_split_seed`).
- Train/test split is stratified by drug × remission and shared across stages via `cache/split_info_healing.json`.
- Stage 2 selects top clinical features by correlation with remission, then fits models with hold-out validation for threshold selection.
- Stage 3 dual-head training enforces the clinical constraint healing = 1 ⟹ remission = 1 via masked loss + constraint terms.
- Absolute local paths are not required; all data/model paths come from the generated config.

## License

See `LICENSE`.
