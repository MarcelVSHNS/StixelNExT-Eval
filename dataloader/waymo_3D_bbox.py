import glob
import os

import numpy as np
import stixel as stx
import tensorflow as tf
from PIL import Image
from waymo_open_dataset import dataset_pb2 as open_dataset


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
    def __init__(self, tf_frame, name: str, cam_idx: int = 0):
        # front = 0, front_left = 1, side_left = 2, front_right = 3, side_right = 4
        self.frame: open_dataset.Frame = tf_frame
        self.cam_idx: int = cam_idx
        self.name = name
        self.bboxes = filter_bboxes_by_hfov(tf_frame.laser_labels, np.deg2rad(50.4))
        img = sorted(tf_frame.images, key=lambda i: i.name)[cam_idx]
        self.image = Image.fromarray(tf.image.decode_jpeg(img.image).numpy())
        front_cam_calib = sorted(self.frame.context.camera_calibrations, key=lambda i: i.name)[0]
        K = self._get_camera_matrix(front_cam_calib.intrinsic)
        T = np.linalg.inv(np.array(front_cam_calib.extrinsic.transform).reshape(4, 4))
        self.calib: stx.stixel_world_pb2.CameraInfo = stx.stixel_world_pb2.CameraInfo()
        self.calib.T.extend(T.flatten().tolist())
        self.calib.K.extend(K.flatten().tolist())
        self.calib.height = self.image.size[1]
        self.ground_truth = None

    @staticmethod
    def _get_camera_matrix(intrinsics):
        # Projection matrix: https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
        # Extract intrinsic parameters: 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]
        waymo_cam_RT = np.array([0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4)
        f_u, f_v, c_u, c_v = intrinsics[:4]
        # Construct the camera matrix K
        K_tmp = np.array([
            [f_u, 0, c_u, 0],
            [0, f_v, c_v, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        camera_mtx = K_tmp @ waymo_cam_RT
        return camera_mtx[:3, :3]


class WaymoDataLoader:
    def __init__(self, data_dir: str, first_only=False):
        self.name: str = "waymo-od"
        self.data_dir = os.path.join(data_dir)
        self.first_only: bool = first_only
        self.img_size = {'width': 1920, 'height': 1280}
        self.record_map = sorted(glob.glob(os.path.join(self.data_dir, '*.tfrecord')))
        self.cam_idx: int = 0
        print(f"Found {len(self.record_map)} tf record files")

    def __getitem__(self, idx):
        frames = self.unpack_single_tfrecord_file_from_path(self.record_map[idx])
        waymo_data_chunk = []
        for frame_num, tf_frame in frames.items():
            # reduce the number of samples by 2
            if frame_num % 2 == 0:
                name: str = f"{tf_frame.context.name}_{frame_num}_{open_dataset.CameraName.Name.Name(self.cam_idx + 1)}"
                waymo_data_chunk.append(WaymoData(tf_frame=tf_frame,
                                                  name=name,
                                                  cam_idx=self.cam_idx))
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
