from typing import Optional

import numpy as np
import stixel as stx


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


def _convert_to_mmdet_bbox(bbox):
    # TODO: convert bbox to CameraInstance3DBoxes (mmdet3d)
    return 0, 1


def calculate_mmdet_bbox_iou(box1, box2):
    box1: CameraInstance3DBoxes = box1
    box2: CameraInstance3DBoxes = box2
    return CameraInstance3DBoxes.overlaps(box1, box2).numpy()[0][0]


def _check_if_bbox_in_bboxes(sample_bbox, bboxes, threshold):
    colors = None
    for bbox in bboxes:
        mmdet_bbox, idx = _convert_to_mmdet_bbox(bbox)
        percentage_inside = calculate_mmdet_bbox_iou(sample_bbox, bbox)
        if percentage_inside >= threshold:
            return 1, colors, idx
    return 0, colors, None


def _get_stixel_range(stixel_coordinates: np.ndarray) -> float:
    ranges = np.sqrt(np.sum(stixel_coordinates ** 2, axis=1))
    mean_range = np.mean(ranges)
    return mean_range

def _get_bbox_range(bbox):
    cx, cy, cz = bbox.box.center_x, bbox.box.center_y, bbox.box.center_z
    range_distance = np.sqrt(cx ** 2 + cy ** 2 + cz ** 2)
    return range_distance


def _count_above_range_threshold(range_score, threshold=30):
    filtered_points = [(range_val, result) for range_val, result in range_score if range_val < threshold]
    count_above_threshold = len(filtered_points)
    result_sum = sum(result for _, result in filtered_points)
    return count_above_threshold, result_sum


def evaluate_sample_3dbbox(gt_bboxes, stx_wrld: Optional[stx.StixelWorld] = None, pred_bboxes = None, iou_thres: int = 0.5, bbox_mode: bool = False):
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
        for p_bbox in pred_bboxes:
            result, colors, idx = _check_if_bbox_in_bboxes(p_bbox, gt_bboxes, iou_thres)
            if idx is not None:
                bbox_dict[idx]['count'] += 1
            score += result
            stixel_pt_list.append(p_bbox.corners)
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
