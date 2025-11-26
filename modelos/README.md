# Modelos Entrenados

Este directorio contiene los modelos de detección de herbívoros africanos entrenados en este proyecto.

## 📦 Modelo incluido en el repositorio

### RF-DETR Small ⭐ (Recomendado)

**Archivo:** `rf-detr-small-animaldet.pth` (491 MB, Git LFS)

- **Arquitectura:** RF-DETR Small con backbone DINOv2
- **Parámetros:** ~50M
- **Entrenamiento:** 2 fases (50 épocas cada una)
  - Fase 1: Aprendizaje de representaciones con parches positivos
  - Fase 2: Refinamiento con Hard Negative Mining
- **Tamaño de parche:** 512×512 px
- **Overlap:** 160 px

**Métricas en conjunto de prueba:**
- F1-Score: `[completar desde notebooks/inference_benchmark.ipynb]`
- Precision: `[completar]`
- Recall: `[completar]`

**Uso recomendado:** Balance óptimo entre velocidad y precisión para producción.

---

## 📥 Otros modelos (Google Drive)

### RF-DETR Nano (Ultraligero)

**Link de descarga:** [Google Drive - RF-DETR Nano](https://drive.google.com/file/d/1s25lRX8nOBKBzhoRHjDfMKF5FujzsEiu/view?usp=sharing)

- **Parámetros:** ~5M
- **Tamaño de archivo:** ~50 MB (estimado)
- **Entrenamiento:** 2 fases
- **Ventaja:** Inferencia ultrarrápida en dispositivos con recursos limitados
- **Desventaja:** Menor precisión que Small y Large

**Uso recomendado:** Edge computing, dispositivos móviles, aplicaciones en tiempo real.

### RF-DETR Large (Máxima precisión)

**Link de descarga:** [Google Drive - RF-DETR Large](https://drive.google.com/file/d/13oJdyGycoFRcSjyhbcS18KLVlIK9zPza/view?usp=sharing)

- **Parámetros:** ~135M
- **Tamaño de archivo:** ~540 MB (estimado)
- **Entrenamiento:** 2 fases con 50 épocas cada una
- **Tamaño de parche:** 560×560 px
- **Ventaja:** Máxima precisión en la detección
- **Desventaja:** Mayor tiempo de inferencia y consumo de memoria

**Uso recomendado:** Análisis offline de alta precisión, investigación, benchmarking.

### HerdNet (Baseline replicado)

**Link de descarga:** [Google Drive - HerdNet](https://drive.google.com/file/d/1qbWC3K17Ck_GIMrXAsH9cweCHTVBNyMn/view?usp=sharing)

- **Arquitectura:** CNN baseline de Delplanque et al. (2022)
- **Parámetros:** ~26M
- **Entrenamiento:** Replicación del paper original con 2 fases
- **Propósito:** Modelo de referencia (baseline) para comparación

**Uso recomendado:** Comparación de resultados, validación del dataset.

---

## 💻 Cómo usar los modelos

### 1. Cargar RF-DETR Small (desde el repositorio)

```python
from rfdetr import RFDETRSmall
import torch

# Cargar modelo
model = RFDETRSmall()
checkpoint = torch.load('modelos/rf-detr-small-animaldet.pth', weights_only=False)
state_dict = checkpoint.get('model', checkpoint.get('ema_model'))

# Reinicializar cabeza de detección con 7 clases (6 especies + background)
num_classes = state_dict['class_embed.weight'].shape[0]
model.model.reinitialize_detection_head(num_classes)

# Cargar pesos
model.model.model.load_state_dict(state_dict, strict=True)
model.model.model.eval()

print(f"✓ Modelo cargado: {num_classes} clases")
```

### 2. Cargar modelos desde Google Drive

```python
from rfdetr import RFDETRNano, RFDETRLarge
import torch

# Descargar desde Google Drive y colocar en modelos/
checkpoint_path = 'modelos/rf-detr-nano-animaldet.pth'  # o rf-detr-large-animaldet.pth

# Para Nano
model = RFDETRNano()
checkpoint = torch.load(checkpoint_path, weights_only=False)
# ... (mismo proceso que Small)

# Para Large
model = RFDETRLarge()
checkpoint = torch.load(checkpoint_path, weights_only=False)
# ... (mismo proceso que Small)
```

### 3. Realizar inferencia sobre imágenes completas

```python
from utils.rf_detr import SimpleStitcher
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

# Configurar stitcher (procesa imágenes en parches)
stitcher = SimpleStitcher(
    model=model.model.model,
    patch_size=512,  # 512 para Small/Nano, 560 para Large
    overlap=0,       # Sin overlap en inferencia
    batch_size=16,
    confidence_threshold=0.5,
    device='cuda',
    label_offset=0,
)

# Transformación de imagen
transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Cargar y procesar imagen
image = Image.open('path/to/image.jpg').convert('RGB')
image_tensor = transform(image=np.array(image))['image']

# Inferencia
detections = stitcher(image_tensor)

print(f"Detecciones encontradas: {len(detections['scores'])}")
print(f"Cajas (x1,y1,x2,y2): {detections['boxes']}")
print(f"Clases: {detections['labels']}")
print(f"Confianzas: {detections['scores']}")
```

Ver ejemplos completos en: `notebooks/inference_benchmark.ipynb`

---

## 📊 Comparación de modelos

| Modelo | Parámetros | Tamaño | Parche | F1-Score* | Velocidad** | Uso |
|--------|-----------|---------|--------|-----------|-------------|-----|
| **RF-DETR Nano** | ~5M | ~50 MB | 512×512 | - | ⚡⚡⚡ Muy rápido | Edge devices |
| **RF-DETR Small** ⭐ | ~50M | 491 MB | 512×512 | - | ⚡⚡ Rápido | Producción |
| **RF-DETR Large** | ~135M | ~540 MB | 560×560 | - | ⚡ Normal | Alta precisión |
| **HerdNet (baseline)** | ~26M | - | Variable | - | ⚡⚡ Rápido | Comparación |

\* Completar métricas desde `notebooks/inference_benchmark.ipynb`  
\*\* Velocidad relativa de inferencia en GPU

---

## 📝 Notas importantes

### Git LFS (Large File Storage)

El modelo `rf-detr-small-animaldet.pth` está almacenado con Git LFS. Al clonar el repositorio:

```bash
# Asegúrate de tener Git LFS instalado
git lfs install

# Clonar el repositorio (descarga automáticamente archivos LFS)
git clone https://github.com/amir1226/animaldet.git
```

### Modelos de Google Drive

Para descargar los modelos adicionales:

1. Haz clic en el link de Google Drive correspondiente
2. Descarga el archivo `.pth`
3. Colócalo en la carpeta `modelos/`

### Formato de los checkpoints

Todos los modelos guardan:
- `'model'` o `'ema_model'`: State dict del modelo
- `'optimizer'`: Estado del optimizador (opcional)
- Metadatos de entrenamiento

### Clases detectadas

Los modelos detectan 6 especies:
1. Bushbuck
2. Duiker
3. Eland
4. Impala
5. Warthog
6. Wildebeest
