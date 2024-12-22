from typing import Optional, Tuple
import numpy as np
import stixel as stx
import open3d as o3d


def _rotate_points(points, heading):
    """
    Vectorized rotation of points in 2D by the given heading (rotation matrix).
    points: (N, 2) array where each row is a point (x, y)
    heading: rotation angle in radians
    """
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rotation_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

    # Apply the rotation matrix to all points
    return points @ rotation_matrix.T  # Using matrix multiplication for all points at once


def _calculate_percentage_and_colors_optimized(point_cloud, bbox, tolerance: float = 1.2):
    """
    Vectorized version of _calculate_percentage_and_colors.
    """
    # Extract point cloud positions (x, y, z)
    points = np.array(point_cloud)  # Assuming point_cloud is list of (x, y, z)
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]

    # Extract bbox parameters
    cx, cy, cz = bbox.box.center_x, bbox.box.center_y, bbox.box.center_z
    length, width, height = bbox.box.length * tolerance, bbox.box.width * tolerance, bbox.box.height * tolerance
    heading = bbox.box.heading

    # Translate points relative to bbox center
    rel_x = px - cx
    rel_y = py - cy
    # Stack relative x, y coordinates for vectorized rotation
    rel_points = np.stack([rel_x, rel_y], axis=1)
    # Rotate points
    rotated_points = _rotate_points(rel_points, -heading)
    rotated_x, rotated_y = rotated_points[:, 0], rotated_points[:, 1]

    # Check bounds vectorized
    in_x_bounds = (-length / 2 <= rotated_x) & (rotated_x <= length / 2)
    in_y_bounds = (-width / 2 <= rotated_y) & (rotated_y <= width / 2)
    in_z_bounds = (-height / 2 <= (pz - cz)) & ((pz - cz) <= height / 2)

    # Combine conditions to find points inside the bbox
    inside_mask = in_x_bounds & in_y_bounds & in_z_bounds
    inside_points = np.sum(inside_mask)
    total_points = len(point_cloud)

    # Generate color labels: green for inside points, red for outside
    color_in = np.array([32, 178, 170]) / 255.0  # turquoise
    color_out = np.array([255, 139, 254]) / 255.0  # pink
    colors = np.where(inside_mask[:, None], color_in, color_out)

    percentage_inside = inside_points / total_points if total_points > 0 else 0
    return percentage_inside, colors, bbox.id


def _check_if_stixel_in_bboxes(point_cloud, bboxes, threshold):
    colors = None
    for bbox in bboxes:
        percentage_inside, colors, idx = _calculate_percentage_and_colors_optimized(point_cloud, bbox)
        if percentage_inside >= threshold:
            return 1, colors, idx
    return 0, colors, None


def _convert_waymo_to_bbox_list(waymo_bboxes) -> Tuple[np.array, list]:
    box_list = []
    box_ids = []
    for bbox in waymo_bboxes:
        cx, cy, cz = bbox.box.center_x, bbox.box.center_y, bbox.box.center_z
        length, width, height = bbox.box.length, bbox.box.width, bbox.box.height
        heading = bbox.box.heading
        box_list.append(np.array([cx, cy, cz, length, width, width, heading]))
        box_ids.append(bbox.id)
    return np.array(box_list), box_ids


def convert_bboxes_to_corner_pts(gt_bbox: np.array):
    """
    Converts gt_bbox (N, 7) [cx, cy, cz, length, width, height, heading]
    into shape (N, 8, 3), with bbox corners.

    :param gt_bbox: numpy.ndarray, (N, 7) [cx, cy, cz, length, width, height, heading]
    :return: numpy.ndarray, (N, 8, 3) with corners of the bbox
    """
    num_boxes = gt_bbox.shape[0]
    corners = np.zeros((num_boxes, 8, 3))  # (N, 8, 3)
    for i, box in enumerate(gt_bbox):
        cx, cy, cz, length, width, height, heading = box
        l, w, h = length / 2, width / 2, height / 2
        local_corners = np.array([
            [-l, -w, -h],
            [ l, -w, -h],
            [ l,  w, -h],
            [-l,  w, -h],
            [-l, -w,  h],
            [ l, -w,  h],
            [ l,  w,  h],
            [-l,  w,  h],
        ])
        rotation_matrix = np.array([
            [np.cos(heading), -np.sin(heading), 0],
            [np.sin(heading),  np.cos(heading), 0],
            [0,                0,               1]
        ])
        global_corners = (rotation_matrix @ local_corners.T).T + np.array([cx, cy, cz])
        corners[i] = global_corners
    return corners


def calculate_iou_3d(corners1, corners2):
    """
    Calculates the IoU (Intersection over Union) between two 3D bounding boxes.

    :param corners1: numpy.ndarray, (8, 3), Corners of the first box
    :param corners2: numpy.ndarray, (8, 3), Corners of the second box
    :return: float, IoU-Value
    """
    min1 = np.min(corners1, axis=0)
    max1 = np.max(corners1, axis=0)
    min2 = np.min(corners2, axis=0)
    max2 = np.max(corners2, axis=0)
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    intersection_dims = np.maximum(intersection_max - intersection_min, 0)
    intersection_volume = np.prod(intersection_dims)
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    union_volume = volume1 + volume2 - intersection_volume
    if union_volume == 0:
        return 0.0
    iou = intersection_volume / union_volume
    return iou


def apply_transformation(bboxes, transformation=None):
    """
    Applies a transformation to a bounding box list.

    :param bboxes: numpy.ndarray, (N, 8, 3), 3D coordinates of the bounding box corners
    :param transformation: numpy.ndarray, (4, 4), Transformation matrix
    :return: numpy.ndarray, (N, 8, 3), Transformed bounding box corners
    """
    if transformation is None:
        # waymo default rotation
        T = np.linalg.inv(np.array([0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4))
    else:
        T = transformation
    num_boxes = bboxes.shape[0]
    transformed_bboxes = np.zeros_like(bboxes)
    for i in range(num_boxes):
        corners_homogeneous = np.hstack((bboxes[i], np.ones((8, 1))))  # (8, 4)
        transformed_corners = (T @ corners_homogeneous.T).T  # (8, 4)
        transformed_bboxes[i] = transformed_corners[:, :3]
    return transformed_bboxes


def _check_if_bbox_in_bboxes(sample_bbox, bboxes, threshold):
    colors = None
    bbox_list, bbox_ids = _convert_waymo_to_bbox_list(bboxes)
    bbox_corner_list = convert_bboxes_to_corner_pts(bbox_list)
    # visu_test([sample_bbox], bbox_corner_list)
    # iou_test = []
    for bbox, idx in zip(bbox_corner_list, bbox_ids):
        percentage_inside = calculate_iou_3d(sample_bbox, bbox)
        # iou_test.append(percentage_inside)
        if percentage_inside >= threshold:
            # print(iou_test)
            return 1, colors, idx
    # print(iou_test)
    return 0, colors, None


def visu_test(prediction, gt_corners):
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Untere Fläche
        [4, 5], [5, 6], [6, 7], [7, 4],  # Obere Fläche
        [0, 4], [1, 5], [2, 6], [3, 7],  # Vertikale Verbindungen
    ]
    prediction_color = [0, 0, 1]  # Blau für prediction
    gt_color = [1, 0, 0]  # Rot für gt_bbox

    # Liste zum Speichern aller geometrischen Objekte
    geometries = []

    # Funktion zum Hinzufügen von Bounding Boxes zu den Geometrien
    def add_bboxes_to_scene(bboxes, color):
        for box in bboxes:
            # Erstelle ein LineSet für jede Bounding Box
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(box),
                lines=o3d.utility.Vector2iVector(lines),
            )

            # Setze die Farbe für die Linien
            line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])

            # Punkte der Bounding Box als PointCloud hinzufügen
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(box)
            point_cloud.paint_uniform_color(color)  # Gleiche Farbe wie Linien

            # Füge LineSet und PointCloud zu den Geometrien hinzu
            geometries.append(line_set)
            geometries.append(point_cloud)

    # Füge beide Gruppen von Bounding Boxes hinzu
    add_bboxes_to_scene(prediction, prediction_color)
    add_bboxes_to_scene(gt_corners, gt_color)
    o3d.visualization.draw_geometries(geometries)


def _get_stixel_range(stixel_coordinates: np.ndarray) -> float:
    ranges = np.sqrt(np.sum(stixel_coordinates ** 2, axis=1))
    mean_range = np.mean(ranges)
    return mean_range


def _get_corner_bbox_range(bbox: np.ndarray) -> np.ndarray:
    center = np.mean(bbox, axis=0)
    distance_to_origin = np.linalg.norm(center)
    return distance_to_origin


def _get_bbox_range(bbox):
    cx, cy, cz = bbox.box.center_x, bbox.box.center_y, bbox.box.center_z
    range_distance = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    return range_distance


def _count_above_range_threshold(range_score, threshold=30):
    filtered_points = [(range_val, result) for range_val, result in range_score if range_val < threshold]
    count_above_threshold = len(filtered_points)
    result_sum = sum(result for _, result in filtered_points)
    return count_above_threshold, result_sum


def evaluate_sample_3dbbox(gt_bboxes, stx_wrld: Optional[stx.StixelWorld] = None, pred_bboxes = None, trans_mtx = None, iou_thres: int = 0.5, bbox_mode: bool = False):
    results = {}
    stixel_pt_list = []
    colors_list = []
    score = 0
    range_score = []
    bbox_dict = {}
    for bbox in gt_bboxes:
        bbox_dict[bbox.id] = {'count': 0,
                              'in_camera': bbox.most_visible_camera_name == 'FRONT',
                              'has_lidar_pts': bbox.num_top_lidar_points_in_box > 2,
                              'is_not_sign': bbox.type != 3,
                              'range': _get_bbox_range(bbox)}
    if bbox_mode:
        assert pred_bboxes is not None
        assert trans_mtx is not None
        waymo_coord_prediction = apply_transformation(pred_bboxes)
        transformed_pred_bboxes = apply_transformation(waymo_coord_prediction, np.linalg.inv(trans_mtx))
        for p_bbox in transformed_pred_bboxes:
            result, colors, idx = _check_if_bbox_in_bboxes(p_bbox, gt_bboxes, iou_thres)
            if not transformed_pred_bboxes.shape[0] == 0:
                range_score.append((_get_corner_bbox_range(p_bbox), result))
            if idx is not None:
                bbox_dict[idx]['count'] += 1
            score += result
            stixel_pt_list.append(p_bbox)
            colors_list.append(colors)

    else:
        assert stx_wrld is not None
        for stxl in stx_wrld.stixel:
            stixel_coord = stx.utils.transformation.convert_stixel_to_points(stxl=stxl,
                                                                             calibration=stx_wrld.context.calibration)
            result, colors, idx = _check_if_stixel_in_bboxes(stixel_coord, gt_bboxes, threshold=iou_thres)
            if not stixel_coord.size == 0:
                range_score.append((_get_stixel_range(stixel_coord), result))
            if idx is not None:
                bbox_dict[idx]['count'] += 1
            score += result
            stixel_pt_list.append(stixel_coord)
            colors_list.append(colors)

    num_bboxes_without_stx = 0
    for bbox in bbox_dict.values():
        # and bbox['in_camera'] is True and bbox['has_lidar_pts'] is True
        if bbox['count'] == 0:
            # count only if the bbox is in the fov of the camera and there are lidar points in the box, else its optional
            num_bboxes_without_stx += 1
    bbox_count_relevant = []
    for bbox in bbox_dict.values():
        # bbox['in_camera'] is True
        if bbox['in_camera'] and bbox['has_lidar_pts'] and bbox['is_not_sign']:
            bbox_count_relevant.append(bbox)
    bbox_score = len(bbox_dict) - num_bboxes_without_stx

    for range in [30, 50]:
        stixel_count, score_range = _count_above_range_threshold(range_score, threshold=range)
        bbox_count_relevant_range = 0
        bbox_score_range = 0
        for bbox in bbox_count_relevant:
            if bbox['range'] < range:
                bbox_count_relevant_range += 1
                if bbox['count'] > 0:
                    bbox_score_range += 1
        if stixel_count != 0:
            results[f'Stixel-Score_{range}'] = score_range / stixel_count
        else:
            results[f'Stixel-Score_{range}'] = 1.0
        if bbox_count_relevant_range != 0:
            results[f'BBox-Score_{range}'] = bbox_score_range / bbox_count_relevant_range
        else:
            results[f'BBox-Score_{range}'] = 1.0

    results['Stixel'] = len(stixel_pt_list)
    results['Points'] = score
    if stixel_pt_list:
        results['Stixel-Score'] = score / len(stixel_pt_list)
    else:
        # if there is no stixel, the prediction from pov-stixel is 100 % correct, no incorrect prediction
        results['Stixel-Score'] = 1.0
    if len(bbox_count_relevant) != 0:
        results['BBox-Score'] = bbox_score / len(bbox_count_relevant)
    else:
        results['BBox-Score'] = 1.0
    results['num_Bbox'] = len(bbox_dict)
    results['num_relevant_Bbox'] = len(bbox_count_relevant)
    results['bbox_points'] = bbox_score
    results['bbox_dist'] = bbox_dict
    return results, stixel_pt_list, colors_list
