# Animal Detection & Classification

Proyecto de detección y clasificación de especies de fauna africana en imágenes aéreas UAV, basado en el dataset de [Delplanque et al. (2022)](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.234).

## 📋 Descripción

Este proyecto implementa y evalúa modelos de deep learning para la detección automática y clasificación de mamíferos africanos en imágenes aéreas de alta resolución capturadas por UAVs (drones). El objetivo es desarrollar una solución robusta que pueda asistir en tareas de monitoreo de fauna silvestre en áreas protegidas.

### Dataset y Objetivo

El dataset proviene de vuelos UAV en el Parque Nacional Virunga (RDC) y reservas de Botsuana, Namibia y Sudáfrica, capturando 6 especies en entornos de bosque tropical, sabana y pastizales:

| Especie | Individuos (Train/Val/Test) | Dificultad |
|---------|----------------------------|------------|
| **Elephant** | 2012 / 264 / 688 | Media (variabilidad de sombras) |
| **Topi** | 1678 / 369 / 675 | Media (grupos densos) |
| **Kob** | 1732 / 161 / 477 | Baja |
| **Buffalo** | 1058 / 102 / 349 | Media (oclusiones) |
| **Warthog** | 316 / 43 / 74 | Alta (tamaño pequeño, <100 ejemplos) |
| **Waterbuck** | 166 / 39 / 36 | Alta (desbalance severo) |
| **Total** | 6,962 / 978 / 2,299 | — |

**Métricas objetivo (baseline HerdNet):**
- F1 Score: **83.5%**
- MAE: 1.9
- RMSE: 3.6
- Accuracy: 92.2%

## 🏗️ Estructura del Proyecto

```
animaldet/
├── animaldet/                    # Paquete principal de Python
│   ├── app/                      # API FastAPI (en desarrollo)
│   ├── data/                     # Módulos de procesamiento de datos
│   │   └── transformers/         # Transformaciones personalizadas
│   ├── inference/                # Módulos de inferencia
│   ├── models/                   # Definiciones de arquitecturas
│   ├── preprocessing/            # Preprocesamiento de imágenes
│   ├── train/                    # Scripts de entrenamiento
│   └── utils/                    # Utilidades compartidas
│
├── experiments/                  # Experimentos y reproducción de papers
│   ├── HerdNet/                  # Reproducción de HerdNet (Delplanque et al.)
│   │   ├── experiment_1/         # Entrenamiento 2-etapas clásico
│   │   │   ├── scripts/
│   │   │   │   ├── train_stage1.py           # Stage 1: Positive patches
│   │   │   │   ├── train_stage2.py           # Stage 2: Hard Negative Patches
│   │   │   │   ├── generate_hnps.py          # Generación HNPs
│   │   │   │   └── predict_evaluate_full_image.py
│   │   │   └── README.md
│   │   ├── experiment_2/         # Variante con mejoras
│   │   │   ├── scripts/
│   │   │   │   ├── 1_train.py
│   │   │   │   ├── 2_inference_for_hard_negatives.py
│   │   │   │   ├── 3_train_over_hnp.py
│   │   │   │   └── 4_eval_test_scores.py
│   │   │   └── README.md
│   │   └── results/              # Resultados, métricas y visualizaciones
│   │       ├── detections.csv
│   │       ├── infer-and-eval.ipynb
│   │       └── train/
│   │           ├── train_graphics.ipynb      # Gráficas de entrenamiento
│   │           ├── wandb_train_loss_*.csv
│   │           └── wandb_f1_score_*.csv
│   │
│   └── RF-DETR/                  # Reproducción de RF-DETR (Roboflow)
│       ├── experiment_1/         # Baseline DETR + refinamiento
│       │   └── scripts/
│       │       ├── 1_train.py
│       │       ├── 2_eval_full_size.py
│       │       ├── 4_inference.py
│       │       └── 5_confidence_vs_f1.py
│       ├── results/              # Resultados RF-DETR
│       │   ├── detections.csv
│       │   └── evaluation.ipynb
│       ├── simple_coco_patcher.py
│       └── README.md
│
├── infra/                        # Infraestructura y deployment (WIP)
│   ├── ansible/                  # Automatización de configuración
│   ├── kubernetes/               # Manifiestos K8s
│   └── scripts/                  # Scripts de deployment
│
├── ui/                           # Frontend web (planificado)
│
├── pyproject.toml                # Configuración del proyecto (uv)
├── uv.lock                       # Lockfile de dependencias
└── README.md                     # Este archivo
```

### Ambientes de Desarrollo

El proyecto utiliza **dependency groups** de uv para gestionar diferentes entornos según el modelo/framework:

#### 1. HerdNet Environment
Para experimentos con HerdNet (PyTorch + AnimalOC):

```bash
# Instalar dependencias de HerdNet
uv sync --group herdnet

# Activar entorno
source .venv/bin/activate
```

Dependencias incluidas:
- `animaloc`: Librería oficial de HerdNet
- PyTorch, torchvision
- OpenCV, albumentations
- wandb (tracking)

#### 2. RF-DETR Environment
Para experimentos con RF-DETR (DETR + Roboflow):

```bash
# Instalar dependencias de RF-DETR
uv sync --group rfdetr

# Activar entorno
source .venv/bin/activate
```

Dependencias incluidas:
- `rfdetr`: Implementación oficial
- Transformers (Hugging Face)
- PyTorch, supervision
- roboflow SDK

#### 3. Ambientes Futuros (Planificados)

```toml
[dependency-groups]
# Producción - API y serving
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

# Infraestructura
infra = [
    "ansible",
    "terraform",
]
```

## 📄 Licencia

Este proyecto utiliza código de:
- **HerdNet/AnimalOC**: MIT License (Alexandre Delplanque)
- **RF-DETR**: Apache 2.0 License (Roboflow)


## 👥 Contacto

Para preguntas sobre el proyecto o colaboraciones, por favor abre un issue en GitHub.

---

**Last Updated:** 2025-10-29  
**Status:** 🟡 En desarrollo activo (experimentos HerdNet completados, RF-DETR en progreso)

