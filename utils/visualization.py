import numpy as np
import open3d as o3d


def _get_bboxes_in_shape(bboxes):
    bounding_boxes = []
    for box in bboxes:
        center_x, center_y, center_z = box.box.center_x, box.box.center_y, box.box.center_z
        length, width, height = box.box.length, box.box.width, box.box.height
        heading = box.box.heading

        box_corners = np.array([
            [-length / 2, -width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, width / 2, height / 2],
            [-length / 2, width / 2, height / 2]
        ])

        # Schritt 2: Rotation (Heading) um die Z-Achse anwenden
        rotation_matrix = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])

        # Eckpunkte drehen und anschließend um das Zentrum verschieben
        rotated_corners = box_corners @ rotation_matrix.T
        rotated_corners += np.array([center_x, center_y, center_z])

        # Schritt 3: Bounding Box in Open3D visualisieren
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Untere Ebene
            [4, 5], [5, 6], [6, 7], [7, 4],  # Obere Ebene
            [0, 4], [1, 5], [2, 6], [3, 7]  # Verbindungen zwischen Ebenen
        ]

        colors = [[0, 0, 0] for i in range(len(lines))]  # Farbe Schwarz

        # Erstelle LineSet für die Box-Visualisierung
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(rotated_corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bounding_boxes.append(line_set)

def draw_stixel_and_bboxes(stixel_pts, colors, bboxes):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(stixel_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))