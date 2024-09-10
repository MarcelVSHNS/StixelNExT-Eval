from dataloader import WaymoDataLoader, WaymoData
from stixel.utils import draw_stixels_on_image
import open3d as o3d
import numpy as np
import yaml




def rotate_point(x, y, heading):
    """Rotiert einen Punkt (x, y) um den Heading-Winkel."""
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    x_new = cos_h * x - sin_h * y
    y_new = sin_h * x + cos_h * y
    return x_new, y_new


def is_point_in_bbox(point, bbox, tol=1.2):
    """Überprüft, ob ein Punkt in einer Bounding Box liegt."""
    # Punkt und Box-Koordinaten relativ zum Zentrum der Box
    px, py, pz = point
    cx, cy, cz  = bbox.center_x, bbox.center_y, bbox.center_z
    length, width, height = bbox.length * tol, bbox.width * tol, bbox.height * tol
    heading = bbox.heading

    # Punkt relativ zum Box-Zentrum verschieben
    rel_x = px - cx
    rel_y = py - cy

    # Punkt um den Heading-Winkel der Box rotieren
    rotated_x, rotated_y = rotate_point(rel_x, rel_y, -heading)

    # Box-Grenzen überprüfen
    in_x_bounds = -length / 2 <= rotated_x <= length / 2
    in_y_bounds = -width / 2 <= rotated_y <= width / 2
    in_z_bounds = -height / 2 <= (pz - cz) <= height / 2

    return in_x_bounds and in_y_bounds and in_z_bounds


def calculate_percentage_and_colors(point_cloud, bbox):
    """Berechnet den Prozentsatz der Punkte, die in der Bounding Box liegen und weist Farben zu."""
    total_points = len(point_cloud)
    inside_points = 0
    colors = []

    for point in point_cloud:
        if is_point_in_bbox(point, bbox.box):
            inside_points += 1
            color = [77, 234, 234]
            colors.append(np.array(color) / 255.0)  # Türkis für Treffer
        else:
            color = [255, 139, 254]
            colors.append(np.array(color) / 255.0)  # Pink für Verfehlungen

    percentage_inside = (inside_points / total_points) * 100
    return percentage_inside, colors, bbox.id


def count_hits(point_cloud, bboxes, threshold=50):
    """Zählt, wie viele Punktwolken einen bestimmten Prozentsatz in der Bounding Box treffen."""
    for bbox in bboxes:
        percentage_inside, colors, idx = calculate_percentage_and_colors(point_cloud, bbox)
        if percentage_inside > 0:
            if percentage_inside >= threshold:
                return 1, colors, idx
            else:
                return 0, colors, idx
    return -1, colors, None


with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
loader = WaymoDataLoader(data_dir=config['metric_data_path'],
                         result_dir=config['results_path'],
                         first_only=True)
sample: WaymoData = loader[0][0]

stixel_pt_list = []
colors_list = []
score = 0
count = 0
bbox_dict = {}
for bbox in sample.laser_labels:
    bbox_dict[bbox.id] = 0
for stixel in sample.stixel_wrld.stixel:
    count += 1
    stixel_coord, _ = stixel.convert_to_pseudo_coordinates(camera_calib=sample.stixel_wrld.camera_info)
    result, colors, idx = count_hits(stixel_coord, sample.laser_labels)
    if idx is not None:
        bbox_dict[idx] +=1
    score += result
    stixel_pt_list.append(stixel_coord)
    colors_list.extend(colors)

num_bboxes_without_stx = 0
for value in bbox_dict.values():
    if value == 0:
        num_bboxes_without_stx += 1
bbox_score = len(bbox_dict) - num_bboxes_without_stx
print(f"Stixel: {len(stixel_pt_list)}")
print(f"Score: {score}")
print(f"Result: {100 / len(stixel_pt_list) * score} %.")
print(f"BBox-TN-Rate: {100 / len(bbox_dict) * bbox_score} % ({bbox_score} out of {len(bbox_dict)}).")

# Visualise point cloud
o3d.visualization.draw_geometries([pcd] + bounding_boxes)
print("Data loaded!")
