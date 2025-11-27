# Detección y Clasificación de Animales

Sistema de detección automática de mamíferos africanos en imágenes aéreas de UAV usando RF-DETR, basado en el dataset de [Delplanque et al. (2022)](https://zslpublications.onlinelibrary.wiley.com/doi/10.1002/rse2.234).

---

## 📖 Documentación Completa

Para la **documentación completa del proyecto** incluyendo arquitectura, guía de usuario detallada y análisis de resultados, por favor consultar:

**[📄 FULL_DOC.md](./FULL_DOC.md)** - Documentación técnica completa del proyecto

---

## 🔗 Enlaces Rápidos

- **Demo en vivo:** [AnimalDet App](http://animaldet-alb-510958915.us-east-1.elb.amazonaws.com/)
- **Documentación completa:** [FULL_DOC.md](./FULL_DOC.md) - Arquitectura, guía de usuario detallada y resultados
- **Guía de instalación:** [INSTALL.md](./INSTALL.md) - Despliegue local y en la nube

## 🚀 Inicio Rápido

## 📋 Description

### Ejecutar la aplicación (Docker)

```bash
# Clonar el repositorio
git clone https://github.com/amir1226/animaldet.git

# Construir y ejecutar
docker build -t animaldet:latest .
docker run -p 8000:8000 animaldet:latest
```

**Acceso:**
- Interfaz Web: http://localhost:8000
- API: http://localhost:8000/api/inference
- Health: http://localhost:8000/health

**Ejemplo de uso de la API:**
```bash
curl -X POST http://localhost:8000/api/inference \
  -H "Content-Type: application/octet-stream" \
  --data-binary @imagen.jpg
```

> **Nota:** La primera construcción tarda ~5-10 minutos (build de frontend + conversión ONNX).

## Despliegue en la nube

Para instrucciones de despliegue en la nube (actualmente soportamos despliegue en AWS ECS) visite nuestra [guia](./INSTALL.md)


## 📊 Dataset

**Fuente:** Parque Nacional Virunga (RDC) y reservas en Botswana, Namibia, Sudáfrica  
**Especies:** 6 clases de herbívoros africanos  
**Resolución:** Imágenes aéreas de 24 MP (6000×4000 px)

| Especie | Train / Val / Test | Dificultad |
|---------|-------------------|------------|
| **Elefante** | 2,012 / 264 / 688 | Media (sombras variables) |
| **Topi** | 1,678 / 369 / 675 | Media (grupos densos) |
| **Kob** | 1,732 / 161 / 477 | Baja |
| **Búfalo** | 1,058 / 102 / 349 | Media (oclusiones) |
| **Facóquero** | 316 / 43 / 74 | Alta (tamaño pequeño) |
| **Cobo de agua** | 166 / 39 / 36 | Alta (desbalance severo) |
| **Total** | **6,962 / 978 / 2,299** | — |

**Descarga:** Ver instrucciones en [`datos/README.md`](datos/README.md)  
**Link:** [Dataverse - Université de Liège](https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0)

## 🎯 Resultados

**Métricas finales en conjunto de prueba** (después de Hard Negative Mining):

| Modelo | Precision | Recall | F1-Score | MAE | RMSE |
|--------|-----------|--------|----------|-----|------|
| HerdNet | 0.8229 | 0.8425 | 0.8326 | 1.90 | 3.67 |
| RF-DETR Nano | 0.8161 | 0.6407 | 0.7178 | 3.73 | 6.90 |
| **RF-DETR Small** ⭐ | **0.9385** | **0.8691** | **0.9024** | **1.15** | **2.41** |
| RF-DETR Large | 0.8893 | 0.8839 | 0.8866 | 1.22 | 3.10 |

**Resumen:**
- **RF-DETR Small**: Mejor F1-Score (90.24%) y menor error de conteo (MAE 1.15)
- **RF-DETR Large**: Mejor recall (88.39%) para máxima recuperación
- **Mejora sobre HerdNet**: +8.4% F1, +39% reducción MAE

**Latencia de inferencia** (NVIDIA A100, imágenes 24MP):
- RF-DETR Small: **193 ms** ⚡ (más rápido)
- RF-DETR Nano: 209 ms
- RF-DETR Large: 418 ms
- HerdNet: 441 ms

## 💡 Enfoque

**Pipeline de dos fases:**
1. **Fase 1:** Entrenamiento inicial sobre parches con animales (alta recuperación)
2. **Fase 2:** Hard Negative Mining (reduce falsos positivos manteniendo recall)

**Ventajas de RF-DETR:**
- **Sin NMS:** Predicción end-to-end de conjuntos de objetos → elimina subconteo en manadas densas
- **Contexto global:** Backbone DINOv2 (ViT-L/14) captura dependencias de largo alcance
- **Mejora en minoritarias:** +19% F1 en Cobo de agua, +25% precisión en Facóquero vs HerdNet

**Stack de despliegue:**
- Exportación a ONNX Runtime para inferencia eficiente
- API FastAPI con microservicios
- Orquestación AWS ECS/Fargate (Terraform)
- UI React/Vite para revisión cualitativa

## 🏗️ Estructura del Proyecto

```
animaldet/
├── datos/                        # Instrucciones de descarga del dataset
│   └── README.md
├── modelos/                      # Modelos entrenados
│   ├── README.md                 # Documentación y links a Google Drive
│   └── rf-detr-small-animaldet.pth  # RF-DETR Small (491 MB, Git LFS)
├── Notebooks/                    # Notebooks de análisis y entrenamiento
│   ├── exploratory_data_analysis.ipynb  # EDA del dataset
│   ├── data_preparation_flow.ipynb      # Pipeline de preprocesamiento
│   ├── herdnet_train.ipynb              # Entrenamiento HerdNet
│   ├── detr_train.ipynb                 # Entrenamiento RF-DETR
│   ├── evaluation.ipynb                 # Comparación de modelos
│   ├── inference_benchmark.ipynb        # Medición de latencias
│   └── utils/                           # Helpers para notebooks
├── animaldet/                    # Paquete Python principal
│   ├── app/                      # API FastAPI
│   ├── inference/                # Inferencia ONNX/PyTorch
│   ├── data/                     # Procesamiento de datos
│   └── utils/                    # Utilidades compartidas
├── experiments/                  # Scripts de reproducción
│   ├── HerdNet/                  # HerdNet (baseline)
│   └── RF-DETR/                  # RF-DETR experiments
├── tools/                        # Herramientas de conversión
│   └── convert_to_onnx.py        # PyTorch → ONNX
├── ui/                           # Frontend React/Vite
├── infra/                        # Infraestructura AWS (Terraform)
├── Dockerfile                    # Build multi-stage (frontend + ONNX + API)
├── Makefile                      # Comandos de automatización
└── pyproject.toml                # Dependencias (uv)
```

## 🔧 Desarrollo

> Esta sección es para **entrenar modelos** o **experimentar**. Si solo quieres usar la aplicación, ve a [Inicio Rápido](#-inicio-rápido).

### Prerequisitos
- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) (gestor de paquetes)
- CUDA 11.8+ (opcional, para GPU)

### Instalación

```bash
# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalar dependencias base
uv sync
```

### Entornos por Framework

El proyecto usa **grupos de dependencias** para separar HerdNet y RF-DETR:

**Para HerdNet:**
```bash
uv sync --group herdnet
source .venv/bin/activate
```

**Para RF-DETR:**
```bash
uv sync --group rfdetr
source .venv/bin/activate
```

Ambos grupos incluyen:
- PyTorch, albumentations, OpenCV
- wandb (tracking de experimentos)
- ipykernel (para notebooks)

## 📚 Notebooks

Los notebooks documentan el flujo completo de experimentación:

| Notebook | Descripción |
|----------|-------------|
| `exploratory_data_analysis.ipynb` | EDA del dataset, estadísticas y distribuciones |
| `data_preparation_flow.ipynb` | Visualización del pipeline de parchificación y augmentación |
| `herdnet_train.ipynb` | Entrenamiento HerdNet (baseline) - Fase 1 y 2 |
| `detr_train.ipynb` | Entrenamiento RF-DETR (Nano/Small/Large) - Fase 1 y 2 |
| `evaluation.ipynb` | Comparación de métricas entre modelos y fases |
| `inference_benchmark.ipynb` | Medición de latencias y throughput |

**Ejecutar notebooks:**
```bash
# Activar entorno
uv sync --group rfdetr
source .venv/bin/activate

# Iniciar Jupyter
jupyter notebook Notebooks/
```

**Documentación detallada:** Ver [`Notebooks/README.md`](Notebooks/README.md) para información completa de cada notebook.

## 🎯 Modelos

Ver documentación completa en [`modelos/README.md`](modelos/README.md)

**En el repositorio (Git LFS):**
- **RF-DETR Small** (491 MB): F1 90.24%, latencia 193ms ⭐

**En Google Drive:**
- **RF-DETR Nano**: ~50 MB, para edge devices
- **RF-DETR Large**: ~540 MB, máxima precisión (F1 88.66%)
- **HerdNet**: Baseline de referencia

**Uso de modelos:**
```python
from rfdetr import RFDETRSmall
import torch

# Cargar modelo
model = RFDETRSmall()
checkpoint = torch.load('modelos/rf-detr-small-animaldet.pth')
model.model.load_state_dict(checkpoint['model'])
```

Ver ejemplos completos en [`modelos/README.md`](modelos/README.md).

## 📄 Licencia

Este proyecto usa código de:
- **HerdNet/AnimalOC**: Licencia MIT (Alexandre Delplanque)
- **RF-DETR**: Licencia Apache 2.0 (Roboflow)

---

## 👥 Equipo

Este proyecto fue desarrollado por:

- **Amir Sadour** - [@amir1226](https://github.com/amir1226)
- **Camilo Rodriguez**
- **Claudia Agudelo**
- **Luis Manrique**

