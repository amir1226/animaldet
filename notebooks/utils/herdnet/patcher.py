import os
import PIL
import torchvision
import numpy
import cv2
import pandas

from albumentations import PadIfNeeded
from tqdm import tqdm

from animaloc.data import ImageToPatches, PatchesBuffer, save_batch_images


def patch_images(
    root: str,
    height: int,
    width: int,
    overlap: int,
    dest: str,
    csv_path: str = None,
    min_visibility: float = 0.1,
    save_all: bool = False
):
    """
    Corta imágenes en parches, opcionalmente usando anotaciones.

    Parámetros
    ----------
    root : str
        Carpeta con imágenes.
    height : int
        Altura de cada parche.
    width : int
        Anchura de cada parche.
    overlap : int
        Solapamiento entre parches.
    dest : str
        Carpeta donde guardar los parches.
    csv_path : str, opcional
        Ruta al CSV con anotaciones.
    min_visibility : float
        Fracción mínima del área para conservar la anotación.
    save_all : bool
        Guardar todos los parches y no solo los anotados.
    """

    os.makedirs(dest, exist_ok=True)
    
    images_paths = [
        os.path.join(root, p)
        for p in os.listdir(root)
        if not p.endswith('.csv')
    ]

    # Si hay anotaciones
    if csv_path is not None:
        patches_buffer = PatchesBuffer(
            csv_path, root, (height, width),
            overlap=overlap,
            min_visibility=min_visibility
        ).buffer

        # Guardar anotaciones de parches
        patches_buffer.drop(columns='limits') \
                      .to_csv(os.path.join(dest, 'gt.csv'), index=False)

        # Si NO se deben guardar todos, sólo procesar imágenes anotadas
        if not save_all:
            images_paths = [
                os.path.join(root, x)
                for x in pandas.read_csv(csv_path)['images'].unique()
            ]

    for img_path in tqdm(images_paths, desc='Exporting patches'):
        pil_img = PIL.Image.open(img_path)
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)

        # Caso con anotaciones
        if csv_path is not None:

            if save_all:
                # Guardar todos los parches
                patches = ImageToPatches(
                    img_tensor, (height, width),
                    overlap=overlap
                ).make_patches()
                save_batch_images(patches, img_name, dest)

            else:
                # Solo parches anotados
                padder = PadIfNeeded(
                    height, width,
                    position=PadIfNeeded.PositionType.TOP_LEFT,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0
                )


                img_ptch_df = patches_buffer[
                    patches_buffer['base_images'] == img_name
                ]

                for row in img_ptch_df[['images', 'limits']].to_numpy().tolist():
                    ptch_name, limits = row[0], row[1]
                    cropped_img = numpy.array(pil_img.crop(limits.get_tuple))
                    padded_img = PIL.Image.fromarray(
                        padder(image=cropped_img)['image']
                    )
                    padded_img.save(os.path.join(dest, ptch_name))

        # Caso sin anotaciones
        else:
            patches = ImageToPatches(
                img_tensor, (height, width), overlap=overlap
            ).make_patches()
            save_batch_images(patches, img_name, dest)
