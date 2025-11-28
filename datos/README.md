# Datos

## 📥 Descarga del Dataset

El dataset utilizado en este proyecto es de acceso público. Hay **dos opciones** según tus necesidades:

### Opción 1: Dataset Procesado (Recomendado) ⭐

**Dataset con anotaciones ya convertidas a puntos centrales**  
🔗 [Google Drive - data.zip](https://drive.google.com/file/d/1CcTAZZJdwrBfCPJtVH6VBU3luGKIN9st/view)

- ✅ Anotaciones CSV con puntos centrales (listas para usar)
- ✅ Parches pre-generados para entrenamiento
- ✅ No requiere preprocesamiento adicional

**Instrucciones:**
```bash
# Descargar desde Google Drive y extraer en la raíz del proyecto
unzip data.zip
# Esto creará la carpeta data-delplanque/
```

**Estructura de `data-delplanque/`:**
```
data-delplanque/
├── train/                    # Imágenes de entrenamiento (24 MP)
├── train.csv                 # Puntos centrales de train
├── train_patches/            # Parches de entrenamiento
├── train_patches.csv         # Puntos centrales de parches
├── val/                      # Imágenes de validación
├── val.csv                   # Puntos centrales de val
├── val_patches/              # Parches de validación
├── test/                     # Imágenes de prueba
└── test.csv                  # Puntos centrales de test
```

### Opción 2: Dataset Original

**Dataset oficial de Alexandre Delplanque et al. (2020)**  
🔗 [Dataverse - Université de Liège](https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0)

- Anotaciones COCO JSON (bounding boxes)
- Requiere conversión a puntos centrales

**Instrucciones:**
```bash
# Descargar desde Dataverse y extraer en la raíz del proyecto
unzip general_dataset.zip
# Esto creará la carpeta general_dataset/
```

**Estructura de `general_dataset/`:**
```
general_dataset/
├── train/                       # Imágenes de entrenamiento (24 MP)
├── val/                         # Imágenes de validación
├── test/                        # Imágenes de prueba
├── train_subframes/             # Parches pre-generados
└── groundtruth/                 # Anotaciones
    ├── json/
    │   ├── big_size/           # Anotaciones COCO (bboxes)
    │   └── sub_frames/         # Anotaciones de parches
    └── csv/                    # Puntos centrales
```

---

## 📊 Resultados de Experimentación

Esta carpeta también contiene los **resultados de inferencias** y **tiempos de ejecución** de los modelos evaluados.

### `detections/`

Predicciones de los modelos en el conjunto de prueba:

**HerdNet:**
- `herdnet_stage1_predictions.csv` - Predicciones Fase 1
- `herdnet_stage2_detections.csv` - Predicciones Fase 2

**RF-DETR Nano:**
- `rfdetr_stage1_detections_points (nano).csv` - Fase 1
- `rfdetr_stage2_detections_points (nano).csv` - Fase 2

**RF-DETR Small:**
- `rfdetr_stage1_detections (small).csv` - Fase 1
- `rfdetr_stage2_detections (small).csv` - Fase 2

**RF-DETR Large:**
- `rfdetr_stage1_detections_points (large).csv` - Fase 1
- `rfdetr_stage2_detections_points (large).csv` - Fase 2

### `latency/`

Tiempos de inferencia medidos en NVIDIA A100:

- `inference_times_herdnet.csv`
- `inference_times_rf_detr_nano.csv`
- `inference_times_rf_detr_small.csv`
- `inference_times_rf_detr_large.csv`

### Formato de los CSV

**CSV de detecciones:**
```csv
image_id,x,y,class_id,confidence
test_001.jpg,1234.5,2345.6,1,0.95
```

**CSV de latencias:**
```csv
image_id,inference_time_ms
test_001.jpg,193.45
```

### Notebooks que generan estos datos

- **`Notebooks/detr_train.ipynb`** - Genera CSV de detecciones de RF-DETR (Nano, Small, Large) para ambas fases
- **`Notebooks/herdnet_train.ipynb`** - Genera CSV de detecciones de HerdNet para ambas fases
- **`Notebooks/inference_benchmark.ipynb`** - Consume detecciones, genera CSV de latencias, calcula métricas

---

## 📚 Citación

Si utilizas este dataset en tu investigación, por favor cita:

```bibtex
@data{Delplanque_2020,
  author = {Delplanque, Alexandre and Foucher, Samuel and Théau, Jérôme and 
            Druoton, Lucie and Lejeune, Philippe and Vermeulen, Cédric},
  publisher = {Université de Liège},
  title = {Multispecies detection and identification of African mammals in 
           aerial imagery using convolutional neural networks},
  year = {2020},
  doi = {10.14428/DVN/CZOXCA},
  url = {https://dataverse.uliege.be/file.xhtml?fileId=11098}
}
```

---

**⚠️ Nota:** Las carpetas `data-delplanque/` y `general_dataset/` están en `.gitignore` y se descargan localmente. Esta carpeta (`datos/`) contiene documentación e instrucciones de descarga, además de los CSV de resultados de experimentación.
