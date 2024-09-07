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
from stixel import StixelWorld


point_dtype = np.dtype([
    ('x', np.float64),
    ('y', np.float64),
    ('z', np.float64),
    ('proj_x', np.int32),
    ('proj_y', np.int32)
])


class WaymoData:
    def __init__(self, tf_frame, name: str, stixel: StixelWorld, cam_idx: int = 0):
        """
        Base class for raw from waymo open dataset
        Args:
            tf_frame:
            stixels:
        """
        super().__init__()
        # front = 0, front_left = 1, side_left = 2, front_right = 3, side_right = 4
        self.frame: open_dataset.Frame = tf_frame
        self.cam_idx: int = cam_idx
        img = sorted(tf_frame.images, key=lambda i: i.name)[cam_idx]
        self.name = name
        self.laser_labels: Label = tf_frame.laser_labels
        self.cam_calib = sorted(self.frame.context.camera_calibrations, key=lambda i: i.name)[0]
        self.stixel_wrld = stixel
        K = self._get_camera_matrix()
        # according to Waymo docs: // Camera frame to vehicle frame.
        T = np.linalg.inv(np.array(self.cam_calib.extrinsic.transform).reshape(4, 4))
        self.stixel_wrld.camera_info.K = K
        self.stixel_wrld.camera_info.T = T
        # self.stixel_pts: np.array = self.stixel_wrld.get_pseudo_coordinates(respect_t=False)
        self.image: np.array = Image.fromarray(tf.image.decode_jpeg(img.image).numpy())
        self._point_slices(self.frame, cam_idx)

    def _get_camera_matrix(self):
        # Extract intrinsic parameters: 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]
        waymo_cam_RT = np.array([0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(4, 4)
        f_u, f_v, c_u, c_v = self.cam_calib.intrinsic[:4]
        # Construct the camera matrix K
        K_tmp = np.array([
            [f_u, 0, c_u, 0],
            [0, f_v, c_v, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        k_exp = K_tmp @ waymo_cam_RT
        return k_exp[:3, :3]

    def _point_slices(self, frame, cam_idx):
        """
        Slices the top lidar point cloud into pieces, fitting for every camera incl. projections
        Args:
            frame: Expects the full frame
        Returns: A list of points and a list of projections: (0 = Front, 1 = Side_left, 2 = Side_right, 3 = Left, 4 = Right)
        shape: [x, y, z, proj_x, proj_y]
        """
        # Cuts for just the front view (front = 0, front_left = 1, side_left = 2, front_right = 3, side_right = 4)
        (range_images, camera_projections, segmentation_labels, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)
        # 3d points in vehicle frame, just pick Top LiDAR
        laser_points = points[0]
        laser_projection_points = tf.constant(cp_points[0], dtype=tf.int32)
        images = sorted(frame.images, key=lambda i: i.name)
        # define mask where the projections equal the picture view
        # (0 = Front, 1 = Side_left, 2 = Side_right, 3 = Left, 4 = Right)
        mask = tf.equal(laser_projection_points[..., 0], images[cam_idx].name)
        # transform points after slicing it from the mask into float values
        laser_points_view = tf.gather_nd(laser_points, tf.where(mask)).numpy()
        laser_camera_projections_view = tf.cast(tf.gather_nd(laser_projection_points, tf.where(mask)), dtype=tf.float32).numpy()
        concatenated_laser_pts = np.column_stack((laser_points_view, laser_camera_projections_view[..., 1:3]))
        self.points = np.array([tuple(row) for row in concatenated_laser_pts], dtype=point_dtype)


class WaymoDataLoader:
    def __init__(self, data_dir: str, result_dir: str, first_only=False):
        """
        Loads a full set of waymo raw in single frames, can be one tf_record file or a folder of  tf_record files.
        provides a list by index for a tfrecord-file which has ~20 frame objects. Every object has lists of
        .images (5 views) and .laser_points (top lidar, divided into 5 fitting views). Like e.g.:
        798 tfrecord-files (selected by "idx")
            ~20 Frames (batch size/ dataset - selected by "frame_num")
                5 .images (camera view - selected by index[])
                5 .laser_points (shape of [..., [x, y, z, img_x, img_y]])
        Args:
            data_dir: specify the location of the tf_records
            additional_views: if True, loads only frames with available camera segmentation
            first_only: doesn't load the full ~20 frames to return a raw sample if True
        """
        super().__init__()
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
                stixel_wrld = StixelWorld.read(stixel_path)
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
