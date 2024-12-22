import numpy as np
import open3d as o3d
import cv2


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

        rotation_matrix = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading), np.cos(heading), 0],
            [0, 0, 1]
        ])

        rotated_corners = box_corners @ rotation_matrix.T
        rotated_corners += np.array([center_x, center_y, center_z])

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        colors = [[0, 0, 0] for i in range(len(lines))]  # Farbe Schwarz

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(rotated_corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bounding_boxes.append(line_set)
    return bounding_boxes

def draw_stixel_and_bboxes(stixel_pts, colors, bboxes):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(stixel_pts))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    bounding_boxes = _get_bboxes_in_shape(bboxes)
    o3d.visualization.draw_geometries([pcd] + bounding_boxes)


def draw_mmdet_bboxes_on_image(image, bboxes, calib, K):
    corners_3d = bboxes.corners  # (N, 8, 3)
    corners_3d_homogeneous = np.concatenate([corners_3d, np.ones((corners_3d.shape[0], 8, 1))],
                                            axis=-1)

    # Transformation matrix (T) from calibration
    #T = np.array(calib.T).reshape(4, 4)
    #K = np.hstack((np.array(calib.K).reshape(3,3), np.array([[0], [0], [0]])))
    # Transform 3D points with the transformation matrix
    #corners_3d_transformed = np.einsum('ij,nmj->nmi', T, corners_3d_homogeneous)
    proj_points = []
    for box in corners_3d_homogeneous:
        box_2d = (K @ box.T).T
        box_2d = box_2d[:, :2] / box_2d[:, 2:3]
        proj_points.append(box_2d)

    for box_2d in proj_points:
        box_2d = box_2d.astype(int)
        for start, end in [(0, 1), (1, 2), (2, 3), (3, 0),
                           (4, 5), (5, 6), (6, 7), (7, 4),
                           (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(image, tuple(box_2d[start]), tuple(box_2d[end]), color=(0, 255, 0), thickness=2)
    return image
