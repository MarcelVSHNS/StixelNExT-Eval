"""
Panoptic idx: [1, 12, 44, 92, 108, 140, 143, 150, 155, 165, 167, 174, 178, 180, 188, 195]
"""
from dataloader import WaymoDataLoader, StixelModel
from typing import List
import yaml
import torch
from torchmetrics import JaccardIndex
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import matplotlib.colors as mcolors
from stixel import Stixel, StixelWorld
import stixel as stx

with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)


def main():
    loader = WaymoDataLoader(data_dir=config["metric_data_path"],
                             first_only=False)
    stxl_model = StixelModel(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    """
    for i in [1, 12, 44, 92, 108, 140, 143, 150, 155, 165, 167, 174, 178, 180, 188, 195]:
        sample = loader[i]
        print(i)
    """
    sample = loader[92]
    sample_sem_seg = sample[0].semantic_label.squeeze()
    reduced_sem_seg = sample_sem_seg[::8, ::8]
    class_set = np.array([2,3,4,5,6,7,8,9,10,11,16])
    binary_array = np.isin(reduced_sem_seg, class_set).astype(int)

    pred_mtx = stxl_model.inference(sample[0].image)
    stx_wrld: StixelWorld = stxl_model.revert(pred_mtx, probability=0.55, calib=sample[0].calib)
    stixel_seg = np.zeros((1280, 1920), dtype=int)
    for stxl in stx_wrld.stixel:
        stixel_seg[stxl.vT:stxl.vB + 1, stxl.u] = 1
    stixel_seg = stixel_seg[::8, ::8]

    iou_metric = JaccardIndex(task="binary")
    pred = torch.from_numpy(stixel_seg)
    target = torch.from_numpy(binary_array)
    iou_score = iou_metric(pred, target)
    print("IoU Score:", iou_score.item())

def evaluate_sample_segmentation(stx_wrld: stx.StixelWorld, segmentation: np.array, image=None):
    if len(stx_wrld.stixel) > 0:
        width = stx_wrld.stixel[0].width
    else:
        width = 8
    # Prepare Target
    reduced_sem_seg = segmentation[::width, ::width]
    # Refer selected classes to: https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/camera_segmentation.proto
    class_set = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16])
    binary_sample_sem_seg = np.isin(reduced_sem_seg, class_set).astype(int)
    target = torch.from_numpy(binary_sample_sem_seg)

    # Prepare Prediction
    stixel_seg = np.zeros((1280, 1920), dtype=int)
    for stxl in stx_wrld.stixel:
        stixel_seg[stxl.vT:stxl.vB + 1, stxl.u] = 1
    reduced_stixel_seg = stixel_seg[::width, ::width]
    pred = torch.from_numpy(reduced_stixel_seg)

    # Metric
    iou_metric = JaccardIndex(task="binary")
    iou_score = iou_metric(pred, target)
    if image:
        seg_img = draw_segmentation(reduced_stixel_seg, binary_sample_sem_seg, image)
        return iou_score.item(), seg_img
    else:
        return iou_score.item()


def draw_segmentation(stixel_seg: np.array, binary_array: np.array, grayscale_image):
    rgb_image = np.array(grayscale_image)[::8, ::8]
    rgb_image = np.stack([rgb_image] * 3, axis=-1)
    binary_rgb_array = np.ones_like(rgb_image, dtype=np.uint8) * 255  # Weißer Hintergrund
    for channel in range(3):  # Für R, G, B-Kanäle
        binary_rgb_array[binary_array == 1, channel] = rgb_image[binary_array == 1, channel]

    # Farb-Matrix erstellen mit RGB und Alpha-Kanal
    combined_image = np.zeros((160, 240, 4), dtype=float)

    # Definiere den Alpha-Wert für die Überlagerung
    alpha = 0.5

    # Grün (binary_array == 1 und stixel_seg == 1)
    green_condition = (binary_array == 1) & (stixel_seg == 1)
    combined_image[green_condition, :3] = mcolors.to_rgb('lightseagreen')
    combined_image[green_condition, 3] = alpha

    # Rot (binary_array == 0 und stixel_seg == 1)
    red_condition = (binary_array == 0) & (stixel_seg == 1)
    combined_image[red_condition, :3] = mcolors.to_rgb('fuchsia')
    combined_image[red_condition, 3] = alpha

    # Alpha-Kanal für restliche Bereiche auf 0 setzen
    other_condition = ~(green_condition | red_condition)
    combined_image[other_condition, 3] = 0

    # Overlay das combined_image auf das binary_rgb_array
    overlay_result = binary_rgb_array.astype(float) / 255.0

    # Alpha Compositing durchführen
    alpha_layer = combined_image[..., 3][..., np.newaxis]
    overlay_result = overlay_result * (1 - alpha_layer) + combined_image[..., :3] * alpha_layer

    # Visualisierung
    plt.figure(figsize=(19, 12))
    plt.imshow(np.clip(overlay_result, 0, 1))
    #plt.title("Segmentation Result")
    plt.axis("off")
    # Speichert das Bild in einem BytesIO-Puffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()  # Schließt die plt-Figur, um Speicherlecks zu vermeiden

    # Wandelt den Puffer in ein PIL-Bild um
    pil_image = Image.open(buf)
    return pil_image


if __name__ == "__main__":
    main()
