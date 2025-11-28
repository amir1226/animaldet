# Notebooks

Esta carpeta contiene los notebooks de Jupyter para **análisis exploratorio**, **entrenamiento de modelos** y **evaluación de resultados**.

## 📓 Notebooks Disponibles

### Notebooks de Análisis

### 1. `exploratory_data_analysis.ipynb`

**Análisis Exploratorio de Datos (EDA)**

- Estadísticas del dataset (distribución de especies, tamaños, etc.)
- Visualización de imágenes y anotaciones
- Análisis de distribución espacial de animales
- Identificación de desbalance de clases

**Cuándo ejecutar:** Antes de entrenar, para entender el dataset

**No requiere:** Modelos entrenados

---

### 2. `data_preparation_flow.ipynb`

**Visualización del Pipeline de Preprocesamiento**

- Muestra cómo se generan los parches desde imágenes completas
- Visualiza el proceso de conversión de bboxes a puntos centrales
- Explica el solapamiento de parches y visibilidad mínima
- Ejemplos de augmentación de datos

**Cuándo ejecutar:** Para entender el preprocesamiento antes de entrenar

**No requiere:** Modelos entrenados

---

### Notebooks de Entrenamiento

### 3. `herdnet_train.ipynb`

**Entrenamiento de HerdNet (Baseline)**

- **Fase 1:** Entrenamiento inicial en parches positivos
- **Generación de HNP:** Hard Negative Patches mining
- **Fase 2:** Refinamiento con HNP
- Evaluación en conjunto de prueba

**Salidas generadas:**
- Modelos entrenados: `outputs/herdnet/stage1/` y `outputs/herdnet/stage2/`
- Detecciones: `datos/detections/herdnet_stage*.csv`

**Prerequisitos:**
- Dataset descargado (`data-delplanque/` o `general_dataset/`)
- Entorno `herdnet`: `uv sync --group herdnet`

---

### 4. `detr_train.ipynb`

**Entrenamiento de RF-DETR (Nano, Small, Large)**

- **Fase 1:** Entrenamiento inicial en parches positivos
- **Generación de HNP:** Hard Negative Patches mining
- **Fase 2:** Refinamiento con HNP
- Evaluación en conjunto de prueba
- Conversión de bboxes a puntos para evaluación

**Salidas generadas:**
- Modelos entrenados: `results/rfdetr_{nano,small,large}/`
- Detecciones: `datos/detections/rfdetr_stage*_detections*.csv`

**Prerequisitos:**
- Dataset descargado (`data-delplanque/` o `general_dataset/`)
- Entorno `rfdetr`: `uv sync --group rfdetr`

---

### Notebooks de Evaluación

### 5. `evaluation.ipynb`

**Evaluación y Comparación de Modelos**

- Carga predicciones de todos los modelos (HerdNet, RF-DETR Nano/Small/Large)
- Compara resultados de Fase 1 vs Fase 2 (antes y después de Hard Negative Mining)
- Calcula métricas por especie: F1, Precision, Recall, MAE, RMSE
- Genera visualizaciones comparativas entre modelos
- Análisis detallado del impacto del Hard Negative Mining

**Salidas generadas:**
- Tablas comparativas de métricas por modelo y fase
- Gráficas de comparación por especie
- Análisis de mejora entre fases

**Prerequisitos:**
- CSV de detecciones en `datos/detections/` (generados por `herdnet_train.ipynb` y `detr_train.ipynb`)
- Ground truth en `data-delplanque/test.csv`

---

### 6. `inference_benchmark.ipynb`

**Medición de Latencias y Benchmark**

- Carga todos los modelos entrenados
- Ejecuta inferencia en el conjunto de prueba
- Mide tiempos de inferencia (latencias)
- Compara velocidad de procesamiento entre modelos
- Genera gráficas de rendimiento

**Salidas generadas:**
- CSV de latencias: `datos/latency/inference_times_*.csv`
- Gráficas de comparación de tiempos
- Análisis de throughput

**Prerequisitos:**
- Modelos entrenados (de `herdnet_train.ipynb` y `detr_train.ipynb`)
- Dataset de prueba

---

## 🔄 Orden de Ejecución Recomendado

### Para entender el dataset:
1. `exploratory_data_analysis.ipynb`
2. `data_preparation_flow.ipynb`

### Para entrenar modelos:
3. `herdnet_train.ipynb` (baseline)
4. `detr_train.ipynb` (RF-DETR variantes)

### Para evaluar y comparar:
5. `evaluation.ipynb` (comparación de métricas)
6. `inference_benchmark.ipynb` (medición de latencias)

## 🚀 Cómo Ejecutar

### 1. Descargar el dataset

Sigue las instrucciones en [`datos/README.md`](../datos/README.md) para descargar:
- **Opción 1 (recomendada):** `data.zip` desde Google Drive
- **Opción 2:** `general_dataset.zip` desde Dataverse

### 2. Instalar dependencias

**Para HerdNet:**
```bash
cd /Users/asadour/Documents/animaldet
uv sync --group herdnet
source .venv/bin/activate
```

**Para RF-DETR:**
```bash
cd /Users/asadour/Documents/animaldet
uv sync --group rfdetr
source .venv/bin/activate
```

### 3. Iniciar Jupyter

```bash
jupyter notebook Notebooks/
```

### 4. Ejecutar los notebooks

Abre el notebook deseado y ejecuta las celdas en orden (Cell → Run All).

## 🛠️ Carpeta `utils/`

Contiene módulos de utilidades reutilizables para los notebooks:

```
utils/
├── common/
│   ├── bbox.py              # Conversión bbox ↔ puntos
├── herdnet/
│   ├── evaluate.py          # Evaluación de HerdNet
│   ├── hnp.py               # Generación de Hard Negative Patches
│   ├── metrics.py           # Cálculo de métricas
│   ├── patcher.py           # Parchificación de imágenes
│   └── train.py             # Entrenamiento de HerdNet
└── rf_detr/
    ├── callbacks.py         # Callbacks de entrenamiento
    ├── detections.py        # Manejo de detecciones
    ├── patcher.py           # Parchificación para RF-DETR
    └── stitcher.py          # Stitching de parches
```

Estos módulos se importan desde los notebooks para evitar duplicación de código.

## 📊 Salidas de los Notebooks

Los notebooks generan varios tipos de salidas:

### Modelos entrenados
- `outputs/herdnet/` - Modelos HerdNet
- `results/rfdetr_nano/` - Modelos RF-DETR Nano
- `results/rfdetr_small/` - Modelos RF-DETR Small
- `results/rfdetr_large/` - Modelos RF-DETR Large

### CSV de resultados
- `datos/detections/` - Predicciones de los modelos (stage 1 y stage 2)
- `datos/latency/` - Tiempos de inferencia

### Análisis y visualizaciones
- Gráficas comparativas de métricas por modelo
- Tablas de evaluación por especie
- Análisis de impacto del Hard Negative Mining

### Datasets procesados
- `data-*-detr/` - Parches generados (Fase 1)
- `data-*-detr-stage2/` - Parches con HNP (Fase 2)

> **Nota:** Las salidas se encuentran en `.gitignore` y se regeneran localmente.

## ⚙️ Configuración

Los notebooks usan configuraciones inline o archivos de configuración mínimos. Parámetros importantes:

**Preprocesamiento:**
- `PATCH_SIZE`: 560 px
- `PATCH_OVERLAP`: 160 px
- `MIN_VISIBILITY`: 0.8

**Entrenamiento:**
- `BATCH_SIZE`: 16 (RF-DETR), 8 (HerdNet)
- `EPOCHS`: 50 (Fase 1 y 2)
- `LEARNING_RATE`: 1e-4 (inicial)

**Evaluación:**
- `CONFIDENCE_THRESHOLD`: 0.5
- `MATCH_RADIUS`: 20 px

## 🐛 Troubleshooting

**Error: ModuleNotFoundError**
```bash
# Reinstalar dependencias del grupo correcto
uv sync --group rfdetr  # o --group herdnet
```

**Error: Dataset no encontrado**
```bash
# Verificar que el dataset esté en la raíz
ls data-delplanque/  # o general_dataset/
```

**Kernel desconectado**
```bash
# Instalar ipykernel en el entorno
uv sync --group rfdetr
source .venv/bin/activate
python -m ipykernel install --user --name=animaldet
```

---

**💡 Tip:** Para experimentación rápida, usa los notebooks de análisis (`exploratory_data_analysis.ipynb`, `data_preparation_flow.ipynb`) sin necesidad de entrenar modelos.

