# Detección y Clasificación de Animales

Proyecto de detección y clasificación de especies de fauna africana en imágenes aéreas de UAV, basado en el dataset de [Delplanque et al. (2022)](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.234).

## 📋 Descripción

Este proyecto implementa y evalúa modelos de aprendizaje profundo para la detección y clasificación automática de mamíferos africanos en imágenes aéreas de alta resolución capturadas por UAVs (drones). El objetivo es desarrollar una solución robusta que pueda asistir en tareas de monitoreo de fauna en áreas protegidas.

### Dataset y Objetivo

El dataset proviene de vuelos de UAV en el Parque Nacional Virunga (RDC) y reservas en Botswana, Namibia y Sudáfrica, capturando 6 especies en entornos de bosque tropical, sabana y pastizales:

| Especie | Individuos (Train/Val/Test) | Dificultad |
|---------|----------------------------|------------|
| **Elefante** | 2012 / 264 / 688 | Media (variabilidad de sombras) |
| **Topi** | 1678 / 369 / 675 | Media (grupos densos) |
| **Kob** | 1732 / 161 / 477 | Baja |
| **Búfalo** | 1058 / 102 / 349 | Media (oclusiones) |
| **Facóquero** | 316 / 43 / 74 | Alta (tamaño pequeño, <100 ejemplos) |
| **Cobo de agua** | 166 / 39 / 36 | Alta (desbalance severo) |
| **Total** | 6,962 / 978 / 2,299 | — |

### Resultados

**Métricas globales en conjunto de prueba** (comparación Fase 1 vs Fase 2):

| Modelo | Fase | Precision | Recall | F1-Score | MAE | RMSE |
|--------|------|-----------|--------|----------|-----|------|
| **HerdNet** | Fase 1 | 0.6154 | 0.8673 | 0.7200 | 4.35 | 9.87 |
| | Fase 2 | 0.8229 | 0.8425 | **0.8326** | 1.90 | 3.67 |
| **RF-DETR Nano** | Fase 1 | 0.5023 | 0.9378 | 0.6542 | 7.74 | 10.93 |
| | Fase 2 | 0.8161 | 0.6407 | 0.7178 | 3.73 | 6.90 |
| **RF-DETR Small** ⭐ | Fase 1 | 0.2615 | 0.9517 | 0.4103 | 23.53 | 32.00 |
| | Fase 2 | **0.9385** | 0.8691 | **0.9024** | **1.15** | **2.41** |
| **RF-DETR Large** | Fase 1 | 0.6133 | 0.9230 | 0.7369 | 4.86 | 7.57 |
| | Fase 2 | 0.8893 | **0.8839** | 0.8866 | 1.22 | 3.10 |

**Mejores resultados (Fase 2):**
- **RF-DETR Small**: F1-Score de **90.24%** con la mejor precision (93.85%) y menor error (MAE 1.15)
- **RF-DETR Large**: Mejor recall (88.39%) para máxima recuperación de individuos
- **HerdNet (baseline)**: F1-Score de 83.26%, estableciendo la línea base de referencia

**Latencia de inferencia** (NVIDIA A100, por imagen 24MP):
- RF-DETR Small: **193 ms** (más rápido)
- RF-DETR Nano: 209 ms
- RF-DETR Large: 418 ms
- HerdNet: 441 ms

### Enfoque General
- **Pipeline de entrenamiento en dos fases:** Fase 1 establece recall sobre parches positivos; Fase 2 inyecta Hard Negative Mining para suprimir falsas alarmas de fondo.
- **Detección basada en Transformers:** RF-DETR elimina Non-Maximum Suppression (NMS) al predecir conjuntos de objetos end-to-end, mitigando el subconteo en manadas densas.
- **Elección del backbone:** Características DINOv2 (ViT-L/14) proveen contexto de largo alcance que demostró ser crítico para elefantes ocluidos y antílopes minoritarios.
- **Stack listo para despliegue:** Los modelos se exportan a ONNX Runtime, se sirven detrás de un microservicio FastAPI, y se orquestan en AWS ECS/Fargate vía Terraform, con una UI React/Vite para revisión cualitativa.

## 🏗️ Estructura del Proyecto

```
animaldet/
├── animaldet/                    # Paquete Python principal
│   ├── app/                      # API FastAPI (en desarrollo)
│   ├── data/                     # Módulos de procesamiento de datos
│   │   └── transformers/         # Transformaciones personalizadas
│   ├── inference/                # Módulos de inferencia
│   ├── models/                   # Definiciones de arquitectura
│   ├── preprocessing/            # Preprocesamiento de imágenes
│   ├── train/                    # Scripts de entrenamiento
│   └── utils/                    # Utilidades compartidas
│
├── datos/                        # Documentación del dataset
│   └── README.md                 # Instrucciones de descarga
│
├── experiments/                  # Experimentos y reproducciones de papers
│   ├── HerdNet/                  # Reproducción de HerdNet (Delplanque et al.)
│   │   ├── experiment_1/         # Entrenamiento clásico en 2 fases
│   │   │   ├── scripts/
│   │   │   │   ├── train_stage1.py           # Fase 1: Parches positivos
│   │   │   │   ├── train_stage2.py           # Fase 2: Hard Negative Patches
│   │   │   │   ├── generate_hnps.py          # Generación de HNPs
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
│       ├── results/              # Resultados de RF-DETR
│       │   ├── detections.csv
│       │   └── evaluation.ipynb
│       ├── simple_coco_patcher.py
│       └── README.md
│
├── infra/                        # Infraestructura y despliegue (WIP)
│   ├── ansible/                  # Automatización de configuración
│   ├── kubernetes/               # Manifiestos K8s
│   └── scripts/                  # Scripts de despliegue
│
├── modelos/                      # Modelos entrenados
│   ├── README.md                 # Documentación de modelos
│   └── rf-detr-small-animaldet.pth  # Modelo RF-DETR Small (Git LFS)
│
├── notebooks/                    # Notebooks de análisis y entrenamiento
│   ├── detr_train.ipynb          # Entrenamiento RF-DETR completo
│   ├── data_preparation_flow.ipynb  # Pipeline de preparación de datos
│   ├── exploratory_data_analysis.ipynb  # Análisis exploratorio
│   ├── inference_benchmark.ipynb # Benchmark de modelos
│   └── utils/                    # Utilidades para notebooks
│
├── ui/                           # Frontend web (planeado)
│
├── pyproject.toml                # Configuración del proyecto (uv)
├── uv.lock                       # Archivo de bloqueo de dependencias
└── README.md                     # Este archivo
```

## 🔧 Instalación y Configuración

### Prerequisitos
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (gestor de paquetes rápido)
- CUDA 11.8+ (para entrenamiento en GPU)

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/amir1226/animaldet.git
cd animaldet

# Instalar uv (si no lo tienes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear entorno e instalar dependencias base
uv sync
```

### Entornos de Desarrollo

El proyecto usa **grupos de dependencias** de uv para gestionar diferentes entornos basados en el modelo/framework:

#### 1. Entorno HerdNet
Para experimentos de HerdNet (PyTorch + AnimalOC):

```bash
# Instalar dependencias de HerdNet
uv sync --group herdnet

# Activar entorno
source .venv/bin/activate
```

Dependencias incluidas:
- `animaloc`: Biblioteca oficial de HerdNet
- PyTorch, torchvision
- OpenCV, albumentations
- wandb (seguimiento)

#### 2. Entorno RF-DETR
Para experimentos de RF-DETR (DETR + Roboflow):

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

#### 3. Entornos Futuros (Planeados)

```toml
[dependency-groups]
# Producción - API y serving
deploy = [
    "fastapi",
    "uvicorn",
    "onnxruntime-gpu",
    "pydantic",
]

# Desarrollo de frontend
ui = [
    "node",  # Vía sistema
]

# Infraestructura
infra = [
    "ansible",
    "terraform",
]
```

## 📊 Dataset

El dataset debe descargarse por separado. Ver instrucciones en [`datos/README.md`](datos/README.md).

**Fuente:** [Université de Liège - Dataverse](https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0)

## 🚀 Uso Rápido

### Entrenamiento RF-DETR

```bash
# Activar entorno RF-DETR
uv sync --group rfdetr
source .venv/bin/activate

# Ver notebook de entrenamiento completo
jupyter notebook notebooks/detr_train.ipynb
```

### Evaluación y Benchmark

```bash
# Ver resultados y métricas
jupyter notebook notebooks/inference_benchmark.ipynb
```

## 📚 Notebooks Principales

- **`notebooks/detr_train.ipynb`**: Entrenamiento completo RF-DETR (2 fases)
- **`notebooks/data_preparation_flow.ipynb`**: Visualización del pipeline de datos
- **`notebooks/exploratory_data_analysis.ipynb`**: Análisis exploratorio del dataset
- **`notebooks/inference_benchmark.ipynb`**: Benchmark de modelos y métricas

## 🎯 Modelos Disponibles

Ver documentación completa en [`modelos/README.md`](modelos/README.md)

### En el repositorio (Git LFS):
- **RF-DETR Small** (491 MB): Mejor balance precisión/velocidad

### En Google Drive:
- **RF-DETR Nano**: Ultraligero para edge devices
- **RF-DETR Large**: Máxima precisión
- **HerdNet**: Baseline de referencia

## 📄 Licencia

Este proyecto usa código de:
- **HerdNet/AnimalOC**: Licencia MIT (Alexandre Delplanque)
- **RF-DETR**: Licencia Apache 2.0 (Roboflow)

---

**Última actualización:** 2025-11-25  
**Estado:** 🟢 RF-DETR Small seleccionado para despliegue; validación ONNX/serving & UI en progreso
