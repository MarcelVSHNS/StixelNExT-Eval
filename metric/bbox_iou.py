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
    box_touched = False
    for bbox in bboxes:
        percentage_inside, colors, idx = _calculate_percentage_and_colors_optimized(point_cloud, bbox)
        if percentage_inside > 0:
            if percentage_inside >= threshold:
                return 1, colors, idx
            else:
                # return 0, colors, idx
                box_touched = True
    if box_touched:
        return 0, colors, idx
    # TODO: should be 0 to enable prec-recall curve
    # return -1, colors, None
    return 0, colors, None


def evaluate_sample_3dbbox(stx_wrld: stx.StixelWorld, bboxes, iou_thres: int = 0.5):
    results = {}
    stixel_pt_list = []
    colors_list = []
    score = 0
    count = 0
    bbox_dict = {}
    for bbox in bboxes:
        bbox_dict[bbox.id] = {'count': 0,
                              'in_camera': bbox.most_visible_camera_name == 'FRONT',
                              'has_lidar_pts': bbox.num_top_lidar_points_in_box > 2}
    for stxl in stx_wrld.stixel:
        count += 1
        stixel_coord = stx.utils.transformation.convert_stixel_to_points(stxl=stxl,
                                                                         calibration=stx_wrld.context.calibration)
        result, colors, idx = _check_if_stixel_in_bboxes(stixel_coord, bboxes, threshold=iou_thres)
        if idx is not None:
            bbox_dict[idx]['count'] += 1
        score += result
        stixel_pt_list.append(stixel_coord)
        colors_list.extend(colors)

    num_bboxes_without_stx = 0
    for bbox in bbox_dict.values():
        # and bbox['in_camera'] is True and bbox['has_lidar_pts'] is True
        if bbox['count'] == 0:
            # count only if the bbox is in the fov of the camera and there are lidar points in the box, else its optional
            num_bboxes_without_stx += 1
    bbox_count_relevant = 0
    for bbox in bbox_dict.values():
        # bbox['in_camera'] is True and
        if bbox['in_camera'] is True and bbox['has_lidar_pts'] is True:
            bbox_count_relevant += 1
    bbox_score = len(bbox_dict) - num_bboxes_without_stx

    results['Stixel'] = len(stixel_pt_list)
    results['Points'] = score
    if stixel_pt_list:
        results['Stixel-Score'] = score / len(stixel_pt_list)
    else:
        # if there is no stixel, the prediction from pov-stixel is 100 % correct, no incorrect prediction
        results['Stixel-Score'] = 1.0
    if bbox_count_relevant != 0:
        results['BBox-Score'] = bbox_score / bbox_count_relevant
    else:
        results['BBox-Score'] = 1.0
    results['num_Bbox'] = len(bbox_dict)
    results['num_relevant_Bbox'] = bbox_count_relevant
    results['bbox_points'] = bbox_score
    results['bbox_dist'] = bbox_dict
    return results, stixel_pt_list, colors_list
