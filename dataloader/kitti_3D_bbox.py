import os
import numpy as np
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid, loadPerspectiveIntrinsic, loadCalibrationCameraToPose


class CameraInfo:
    """
    Class to store camera information.
    Attributes:
        extrinsic (Pose): The extrinsic pose of the camera.
        K (np.array): The camera matrix.
        P (np.array): The projection matrix.
        R (np.array): The rectification matrix.
    Methods:
        __init__(self, xyz: np.array, rpy: np.array, camera_mtx: np.array, projection_mtx: np.array,
            rectification_mtx: np.array):
            Initializes the CameraInformation object with the given camera information.
    """
    def __init__(self, camera_mtx: np.array = np.zeros((3,3)),
                 trans_mtx: np.array = np.zeros((4,4)),
                 proj_mtx: np.array = np.zeros((3,4)),
                 rect_mtx: np.array = np.eye(4)):
        self.K = camera_mtx
        self.T = trans_mtx
        self.P = proj_mtx
        self.R = rect_mtx
        self.extrinsic = self.Pose(xyz=self.T[:3, 3],
                                    rpy=self.calc_euler_angle_from_trans_mtx(self.T))

    class Pose:
        """ Initializes a new Pose object.
        Args:
            xyz (np.array): The position vector in x, y, and z coordinates.
            rpy (np.array): The orientation vector in roll, pitch, and yaw angles. """
        def __init__(self, xyz: np.array, rpy: np.array):
            self.xyz: np.array = xyz
            self.rpy: np.array = rpy

    @staticmethod
    def calc_euler_angle_from_trans_mtx(trans_mtx: np.array):
        """
        Returns the euler angle (roll, pitch, yaw) of the camera matrix from the given transformation matrix.
        """
        rota_mtx = trans_mtx[:3, :3]  # rotation matrix
        yaw = np.arctan2(rota_mtx[1, 0], rota_mtx[0, 0])
        pitch = np.arctan2(-rota_mtx[2, 0], np.sqrt(rota_mtx[2, 1] ** 2 + rota_mtx[2, 2] ** 2))
        roll = np.arctan2(rota_mtx[2, 1], rota_mtx[2, 2])
        return np.array([roll, pitch, yaw])


def load_calib(kitti360Path: str):
    fileCameraToVelo = os.path.join(kitti360Path, 'calibration', 'calib_cam_to_velo.txt')
    Tr = loadCalibrationRigid(fileCameraToVelo)
    print('Loaded %s' % fileCameraToVelo)
    print(Tr.shape)

    fileSickToVelo = os.path.join(kitti360Path, 'calibration', 'calib_sick_to_velo.txt')
    Tr = loadCalibrationRigid(fileSickToVelo)
    print('Loaded %s' % fileSickToVelo)
    print(Tr)

    filePersIntrinsic = os.path.join(kitti360Path, 'calibration', 'perspective.txt')
    Tr = loadPerspectiveIntrinsic(filePersIntrinsic)
    print('Loaded %s' % filePersIntrinsic)
    print(Tr)


if __name__=='__main__':
    dataset_path = '/media/marcel/Data1/Raw/KITTI-360'
    load_calib(dataset_path)
