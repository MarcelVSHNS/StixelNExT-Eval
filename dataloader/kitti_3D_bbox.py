import os
import numpy as np
from kitti360scripts.devkits.commons.loadCalibration import loadCalibrationRigid, loadPerspectiveIntrinsic, loadCalibrationCameraToPose
from stixel import CameraInfo

# calib
def parse_calibration_file(file_path):
    calib_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.split(':', 1)
            calib_data[key.strip()] = value.strip()
    return calib_data

def load_calib(kitti360path: str):
    """ cam is reference with 0,0,0 0,0,0. Provides camera intrinsics (and velo/sick extrinsic) """
    cam2velo = loadCalibrationRigid(os.path.join(kitti360path, 'calibration', 'calib_cam_to_velo.txt'))
    sick2velo = loadCalibrationRigid(os.path.join(kitti360path, 'calibration', 'calib_sick_to_velo.txt'))
    intrinsics = parse_calibration_file(os.path.join(kitti360path, 'calibration', 'perspective.txt'))
    T_velo2cam = np.linalg.inv(cam2velo)
    T_sick2cam = T_velo2cam @ sick2velo

    k_00 = np.array(intrinsics['K_00'].split())
    cam_info = CameraInfo(cam_mtx_k=k_00.reshape(3, 3),
                          reference="self")
    # print(cam_info.__dict__)
    return cam_info #T_sick2cam, T_velo2cam

# 3Dbbox
# TODO: read 3d bboxes

# image
def readfile_structure(root_path: str):
    # TODO: read image_00
    pass



if __name__=='__main__':
    dataset_path = '/media/marcel/Data1/Raw/KITTI-360'
    load_calib(dataset_path)
