import numpy as np
import os
import glob
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.v2 import convert_range_image_to_point_cloud
from waymo_open_dataset.label_pb2 import Label
from waymo_open_dataset.utils import frame_utils
from PIL import Image
import tensorflow as tf
from typing import List
import stixel as stx


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return r, theta, phi


def is_bbox_in_hfov(bbox_center, hfov):
    x, y, z = bbox_center
    _, theta, _ = cartesian_to_spherical(x, y, z)
    half_hfov = hfov / 2
    if -half_hfov <= theta <= half_hfov:
        return True
    return False


def filter_bboxes_by_hfov(bboxes, hfov, fov_margin=8):
    filtered_bboxes = []
    for bbox in bboxes:
        bbox_center = np.array([bbox.box.center_x, bbox.box.center_y, bbox.box.center_z, 1.0])
        transformations_mtx = np.array([
            [1, 0, 0, 1.0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0]
        ])
        transformed_center = np.dot(transformations_mtx, bbox_center)
        if is_bbox_in_hfov(transformed_center[:3], hfov + np.deg2rad(fov_margin)):
            filtered_bboxes.append(bbox)
    return filtered_bboxes


class WaymoData:
    def __init__(self, tf_frame, name: str, stixel: stx.StixelWorld, cam_idx: int = 0):
        # front = 0, front_left = 1, side_left = 2, front_right = 3, side_right = 4
        self.frame: open_dataset.Frame = tf_frame
        self.cam_idx: int = cam_idx
        self.name = name
        self.stxl_wrld = stixel
        self.bboxes = filter_bboxes_by_hfov(tf_frame.laser_labels, np.deg2rad(50.4))


class WaymoDataLoader:
    def __init__(self, data_dir: str, result_dir: str, first_only=False):
        self.name: str = "waymo-od"
        self.data_dir = os.path.join(data_dir)
        self.result_dir = os.path.join(result_dir)
        self.first_only: bool = first_only
        self.img_size = {'width': 1920, 'height': 1280}
        self.record_map = sorted(glob.glob(os.path.join(self.data_dir, '*.tfrecord')))
        print(f"Found {len(self.record_map)} tf record files")

    def __getitem__(self, idx):
        frames = self.unpack_single_tfrecord_file_from_path(self.record_map[idx])
        waymo_data_chunk = []
        for frame_num, tf_frame in frames.items():
            cam_idx: int = 0
            name: str = f"{tf_frame.context.name}_{frame_num}_{open_dataset.CameraName.Name.Name(cam_idx + 1)}"
            stixel_path = os.path.join(self.result_dir, name + ".stx1")
            try:
                stixel_wrld = stx.read(stixel_path)
            except:
                raise FileNotFoundError(f"Stixel: {stixel_path} not found.")
            waymo_data_chunk.append(WaymoData(tf_frame=tf_frame,
                                              name=name,
                                              stixel=stixel_wrld,
                                              cam_idx=cam_idx))
            if self.first_only:
                break
        return waymo_data_chunk

    def __len__(self):
        return len(self.record_map)

    @staticmethod
    def unpack_single_tfrecord_file_from_path(tf_record_filename):
        """ Loads a tf-record file from the given path. Picks only every tenth frame to reduce the dataset and increase
        the diversity of it. With camera_segmentation_only = True, every availabe frame is picked (every tenth is annotated)
        Args:
            tf_record_filename: full path and name of the tf_record file to open
        Returns: a list of frames from the file
        """
        dataset = tf.data.TFRecordDataset(tf_record_filename, compression_type='')
        frame_list = {}
        frame_num = 0
        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if frame.lasers[0].ri_return1.segmentation_label_compressed:
                frame_list[frame_num] = frame
            frame_num += 1
        return frame_list
