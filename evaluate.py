from dataloader import WaymoDataLoader, WaymoData
import open3d as o3d
import numpy as np
import yaml


with open('config.yaml') as yaml_file:
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)
loader = WaymoDataLoader(data_dir=config['metric_data_path'],
                         result_dir=config['results_path'],
                         first_only=False)
sample: WaymoData = loader[0][-1]

point_cloud = o3d.geometry.PointCloud()
stxl_wrld_pts, colors = sample.stixel_wrld.get_pseudo_coordinates(respect_t=True)
point_cloud.points = o3d.utility.Vector3dVector(stxl_wrld_pts)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

bounding_boxes = []
for box in sample.laser_labels:
    center_x, center_y, center_z = box.camera_synced_box.center_x, box.camera_synced_box.center_y, box.camera_synced_box.center_z
    length, width, height = box.camera_synced_box.length, box.camera_synced_box.width, box.camera_synced_box.height
    heading = box.camera_synced_box.heading

    box_corners = np.array([
        [-length / 2, -width / 2, -height / 2],
        [ length / 2, -width / 2, -height / 2],
        [ length / 2,  width / 2, -height / 2],
        [-length / 2,  width / 2, -height / 2],
        [-length / 2, -width / 2,  height / 2],
        [ length / 2, -width / 2,  height / 2],
        [ length / 2,  width / 2,  height / 2],
        [-length / 2,  width / 2,  height / 2]
    ])

    # Schritt 2: Rotation (Heading) um die Z-Achse anwenden
    rotation_matrix = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading),  np.cos(heading), 0],
        [0,               0,                1]
    ])

    # Eckpunkte drehen und anschließend um das Zentrum verschieben
    rotated_corners = box_corners @ rotation_matrix.T
    rotated_corners += np.array([center_x, center_y, center_z])

    # Schritt 3: Bounding Box in Open3D visualisieren
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Untere Ebene
        [4, 5], [5, 6], [6, 7], [7, 4],  # Obere Ebene
        [0, 4], [1, 5], [2, 6], [3, 7]   # Verbindungen zwischen Ebenen
    ]

    colors = [[1, 0, 0] for i in range(len(lines))]  # Farbe Rot

    # Erstelle LineSet für die Box-Visualisierung
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(rotated_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    bounding_boxes.append(line_set)



# Visualise point cloud
o3d.visualization.draw_geometries([point_cloud] + bounding_boxes)
print("Data loaded!")