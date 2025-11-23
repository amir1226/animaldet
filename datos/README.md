# Dataset

## 📥 Descarga del Dataset

El dataset utilizado en este proyecto es de acceso público y debe descargarse antes de ejecutar los notebooks de entrenamiento.

**Dataset:** Alexandre Delplanque et al. (2020) - African Wildlife Detection Dataset  
**Fuente:** Université de Liège - Dataverse  
**Link de descarga:** 🔗 https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0

### Instrucciones de descarga

1. **Descargar el archivo:**
   - Visita el enlace anterior
   - Descarga el archivo `general_dataset.zip` (12.3 GB)

2. **Extraer en la raíz del proyecto:**
   ```bash
   # Desde la raíz del proyecto
   unzip general_dataset.zip
   # Esto creará la carpeta general_dataset/
   ```

3. **Estructura esperada después de la descarga:**
   ```
   general_dataset/
   ├── train/                       # Imágenes de entrenamiento (24 MP, 6000×4000)
   ├── val/                         # Imágenes de validación (24 MP)
   ├── test/                        # Imágenes de prueba (24 MP)
   ├── train_subframes/             # Parches de entrenamiento pre-generados
   └── groundtruth/                 # Anotaciones
       ├── json/
       │   ├── big_size/           # Anotaciones COCO para imágenes completas
       │   │   ├── train_big_size_A_B_E_K_WH_WB.json
       │   │   ├── val_big_size_A_B_E_K_WH_WB.json
       │   │   └── test_big_size_A_B_E_K_WH_WB.json
       │   └── sub_frames/         # Anotaciones COCO para parches
       │       ├── train_subframes_A_B_E_K_WH_WB.json
       │       ├── val_subframes_A_B_E_K_WH_WB.json
       │       └── test_subframes_A_B_E_K_WH_WB.json
       └── csv/                    # Anotaciones en formato CSV (puntos)
           ├── train_big_size_A_B_E_K_WH_WB.csv
           ├── val_big_size_A_B_E_K_WH_WB.csv
           └── test_big_size_A_B_E_K_WH_WB.csv
   ```
   
   **Nota:** `A_B_E_K_WH_WB` representa las iniciales de las 6 especies:
   - **A**ntelope (Bushbuck)
   - **B**ushbuck (Duiker)  
   - **E**land
   - **K**ob (Impala)
   - **WH**arthog
   - **WB**eest (Wildebeest)

## 📊 Descripción del Dataset

### Contenido
- **Tipo:** Imágenes aéreas de herbívoros africanos
- **Especies:** 6 clases
  - Bushbuck
  - Duiker
  - Eland
  - Impala
  - Warthog
  - Wildebeest
- **Resolución:** 6000×4000 píxeles (24 MP)
- **Formato:** JPEG

### Estadísticas
- **Entrenamiento:** ~928 imágenes
- **Validación:** ~232 imágenes  
- **Prueba:** ~258 imágenes

### Formato de anotaciones original

**COCO JSON con bounding boxes:**
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],  # Formato original: bounding boxes
      "area": 12345,
      "iscrowd": 0
    }
  ],
  "categories": [...]
}
```

## 🔄 Preprocesamiento aplicado

Este proyecto convierte las anotaciones originales (bounding boxes) a **puntos centrales (centroides)** para el entrenamiento:

### 1. Conversión bbox → punto central
```python
# De bounding box [x, y, w, h] a punto central
center_x = x + w/2
center_y = y + h/2
```

### 2. Parchificación (Patching)
- **Tamaño de parche:** Variable según modelo (ej. 560×560 px para RF-DETR Large)
- **Solapamiento:** 160 píxeles (basado en el tamaño del individuo más grande)
- **Ajuste de coordenadas:** Conversión de coordenadas globales a locales del parche

### 3. Entrenamiento en dos fases
- **Fase 1:** Solo parches que contienen animales (ejemplos positivos)
- **Fase 2:** Parches con animales + Hard Negative Patches (fondos confusos sin animales)

Ver detalles del pipeline completo en: `notebooks/data_preparation_flow.ipynb`

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
