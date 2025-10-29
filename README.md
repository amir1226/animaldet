# Animal Detection & Classification

Detection and classification project for African wildlife species in aerial UAV images, based on the [Delplanque et al. (2022)](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.234) dataset.

## 📋 Description

This project implements and evaluates deep learning models for automatic detection and classification of African mammals in high-resolution aerial images captured by UAVs (drones). The goal is to develop a robust solution that can assist in wildlife monitoring tasks in protected areas.

### Dataset and Objective

The dataset comes from UAV flights in Virunga National Park (DRC) and reserves in Botswana, Namibia, and South Africa, capturing 6 species in tropical forest, savanna, and grassland environments:

| Species | Individuals (Train/Val/Test) | Difficulty |
|---------|----------------------------|------------|
| **Elephant** | 2012 / 264 / 688 | Medium (shadow variability) |
| **Topi** | 1678 / 369 / 675 | Medium (dense groups) |
| **Kob** | 1732 / 161 / 477 | Low |
| **Buffalo** | 1058 / 102 / 349 | Medium (occlusions) |
| **Warthog** | 316 / 43 / 74 | High (small size, <100 examples) |
| **Waterbuck** | 166 / 39 / 36 | High (severe imbalance) |
| **Total** | 6,962 / 978 / 2,299 | — |

**Target metrics (HerdNet baseline):**
- F1 Score: **83.5%**
- MAE: 1.9
- RMSE: 3.6
- Accuracy: 92.2%

## 🏗️ Project Structure

```
animaldet/
├── animaldet/                    # Main Python package
│   ├── app/                      # FastAPI API (under development)
│   ├── data/                     # Data processing modules
│   │   └── transformers/         # Custom transformations
│   ├── inference/                # Inference modules
│   ├── models/                   # Architecture definitions
│   ├── preprocessing/            # Image preprocessing
│   ├── train/                    # Training scripts
│   └── utils/                    # Shared utilities
│
├── experiments/                  # Experiments and paper reproductions
│   ├── HerdNet/                  # HerdNet reproduction (Delplanque et al.)
│   │   ├── experiment_1/         # Classic 2-stage training
│   │   │   ├── scripts/
│   │   │   │   ├── train_stage1.py           # Stage 1: Positive patches
│   │   │   │   ├── train_stage2.py           # Stage 2: Hard Negative Patches
│   │   │   │   ├── generate_hnps.py          # HNPs generation
│   │   │   │   └── predict_evaluate_full_image.py
│   │   │   └── README.md
│   │   ├── experiment_2/         # Variant with improvements
│   │   │   ├── scripts/
│   │   │   │   ├── 1_train.py
│   │   │   │   ├── 2_inference_for_hard_negatives.py
│   │   │   │   ├── 3_train_over_hnp.py
│   │   │   │   └── 4_eval_test_scores.py
│   │   │   └── README.md
│   │   └── results/              # Results, metrics and visualizations
│   │       ├── detections.csv
│   │       ├── infer-and-eval.ipynb
│   │       └── train/
│   │           ├── train_graphics.ipynb      # Training plots
│   │           ├── wandb_train_loss_*.csv
│   │           └── wandb_f1_score_*.csv
│   │
│   └── RF-DETR/                  # RF-DETR reproduction (Roboflow)
│       ├── experiment_1/         # DETR baseline + refinement
│       │   └── scripts/
│       │       ├── 1_train.py
│       │       ├── 2_eval_full_size.py
│       │       ├── 4_inference.py
│       │       └── 5_confidence_vs_f1.py
│       ├── results/              # RF-DETR results
│       │   ├── detections.csv
│       │   └── evaluation.ipynb
│       ├── simple_coco_patcher.py
│       └── README.md
│
├── infra/                        # Infrastructure and deployment (WIP)
│   ├── ansible/                  # Configuration automation
│   ├── kubernetes/               # K8s manifests
│   └── scripts/                  # Deployment scripts
│
├── ui/                           # Web frontend (planned)
│
├── pyproject.toml                # Project configuration (uv)
├── uv.lock                       # Dependency lockfile
└── README.md                     # This file
```

## 🔧 Installation and Setup

### Prerequisites
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (fast package manager)
- CUDA 11.8+ (for GPU training)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd animaldet

# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install base dependencies
uv sync
```

### Development Environments

The project uses uv **dependency groups** to manage different environments based on the model/framework:

#### 1. HerdNet Environment
For HerdNet experiments (PyTorch + AnimalOC):

```bash
# Install HerdNet dependencies
uv sync --group herdnet

# Activate environment
source .venv/bin/activate
```

Included dependencies:
- `animaloc`: Official HerdNet library
- PyTorch, torchvision
- OpenCV, albumentations
- wandb (tracking)

#### 2. RF-DETR Environment
For RF-DETR experiments (DETR + Roboflow):

```bash
# Install RF-DETR dependencies
uv sync --group rfdetr

# Activate environment
source .venv/bin/activate
```

Included dependencies:
- `rfdetr`: Official implementation
- Transformers (Hugging Face)
- PyTorch, supervision
- roboflow SDK

#### 3. Future Environments (Planned)

```toml
[dependency-groups]
# Production - API and serving
deploy = [
    "fastapi",
    "uvicorn",
    "onnxruntime-gpu",
    "pydantic",
]

# Frontend development
ui = [
    "node",  # Via system
]

# Infrastructure
infra = [
    "ansible",
    "terraform",
]
```

## 📄 License

This project uses code from:
- **HerdNet/AnimalOC**: MIT License (Alexandre Delplanque)
- **RF-DETR**: Apache 2.0 License (Roboflow)

## 👥 Contact

For questions about the project or collaborations, please open an issue on GitHub.

---

**Last Updated:** 2025-10-27
**Status:** 🟡 Actively in development (HerdNet experiments completed, RF-DETR in progress)

