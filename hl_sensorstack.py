#------------------------------------------------------------------------------
# hl_sensorstack.py
# 
# This file should have code representing the code to move between sensors
# in the entire hololens sensor stack. There's a lot of projection/unprojection
# needed to be done. This class should have all of that processing done.
#------------------------------------------------------------------------------
import os
import sys

from hl2ss_stream import HOST_IP, ROOT_PATH

sys.path.append(os.path.join(ROOT_PATH,'hl2ss','viewer'))
import numpy as np

CALIB_PATH = os.path.join(ROOT_PATH,'hl2ss','calibration')

import hl2ss_3dcv

import hl2ss

'''
Sensor Stack Description:
--------------------------
    -> VLC Cameras (30 fps) (grayscale) (640x480)
        -> two forward facing cameras VLC_LF (left front) VLC_RF (right front)
        -> 2 side-view cameras VLC_LL (left left) VLC_RR (right right)
    
    -> Depth (5 fps) (320x288)
        -> Longthrow depth
    -> Research mode IMU 
        -> Accelerometer (m/s^2)
        -> Gyroscope (deg/s)
        -> Magnetometer
    
    -> PV camera (30fps)
'''
class HololensSensorStack:
    def __init__(self, kin_chain):
        self.calib_lt = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, os.path.join(CALIB_PATH,"rm_depth_longthrow"))
        uv2xy = hl2ss_3dcv.compute_uv2xy(self.calib_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        self.xy1, self.lt_scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, self.calib_lt.scale)

        self.calib_lf = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_VLC_LEFTFRONT, os.path.join(CALIB_PATH,"rm_vlc_leftfront"))
        self.calib_rf = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, os.path.join(CALIB_PATH,"rm_vlc_rightfront"))
        self.kin_chain = kin_chain

        
    def update_pv_calibration(self, pv_intrinsics, pv_extrinsics):
        pass

    def undistort(self, undistort_map, data):
        pass
    
    def project_onto_depth_frame(self, pts3, transform):
        '''
        Description:
        ------------
        Project the 3d bbox points onto the depth camera.
        * This function is only for projecting 3d pts from rgb
          to depth

        Params:
        -------
        bbox_pts: np.ndarray of shape (3,N) 
        ''' 
        depth_intrinsics = self.kin_chain.calib_info["depth"].intrinsics
        print("Depth Intrinsics:")
        print(depth_intrinsics)
        return self._compute_transformed_pts(pts3, transform, depth_intrinsics)

    def project_onto_vlc_sensor(self, pts3, transform, vlc_side):
        '''
        Description:
        ------------
        Project the 3d bbox points onto the depth camera.
        * This function is only for projecting 3d pts from depth 
          to either of the vlc cameras 

        Params:
        -------
        bbox_pts: np.ndarray of shape (3,N) 
        ''' 
        if vlc_side not in {"left", "right"}:
            raise Exception("You naughty coder, you chose an invalid side")
        vlc_intrinsics = self.kin_chain.calib_info[f"vlc_{vlc_side}"].intrinsics
        return self._compute_transformed_pts(pts3, transform, vlc_intrinsics)

    def _compute_transformed_pts(self, pts3, transform, intrinsics):
        pts3 = np.vstack((pts3, np.ones((pts3.shape[1],))))
        transformed_pts3 = transform @ pts3
        cam_pts2 = intrinsics @ transformed_pts3
        cam_pts2 = cam_pts2[0:2,:]/cam_pts2[2,:]
        return cam_pts2

class SensorCalibration:
    '''
        WORLD FRAME: frame where map and 3d stuff lays in
        BODY FRAME: frame of the hololens (sensor stack) its origin is fixed and all sensors are relative to it.
        SENSOR FRAME: everything is relative to sensor. normally depth images are in this frame when unprojected.
    '''
    def __init__(self, intrinsics, extrinsics):
        '''
            intrinsics: (4x4)
            extrinsics: (4x4)
        '''
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def get_intrinsics(self):
        return self.intrinsics
    def get_extrinsics(self):
        return self.extrinsics
    
    def sensor2body(self):
        return np.linalg.inv(self.extrinsics)
    def body2sensor(self):
        return self.extrinsics
    
    def body2world(self, pose):
        return pose
    def world2body(self, pose):
        return np.linalg.inv(pose)
    
    def world2sensor(self, pose):
        return self.body2sensor() @ self.world2body(pose)
    def sensor2world(self, pose):
        return self.body2world(pose) @ self.sensor2body()

def get_calibration_from_folder(folder):
    calib_lf = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_VLC_LEFTFRONT, os.path.join(folder,"rm_vlc_leftfront"))
    calib_rf = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, os.path.join(folder,"rm_vlc_rightfront"))
    calib_depth = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, os.path.join(folder,"rm_depth_longthrow"))

    sensor_calib_lf = SensorCalibration(calib_lf.intrinsics.T, calib_lf.extrinsics.T)
    sensor_calib_rf = SensorCalibration(calib_rf.intrinsics.T, calib_rf.extrinsics.T)
    sensor_calib_depth = SensorCalibration(calib_depth.intrinsics.T, calib_depth.extrinsics.T)
    return sensor_calib_lf, sensor_calib_rf, sensor_calib_depth

class KinematicChain:
    '''
    Automate finding the transformation from one frame to another frame
    '''
    def __init__(self, rgb_intrinsics, rgb_extrinsics):
        s1, s2, s3 = get_calibration_from_folder(CALIB_PATH)
        self.valid_frames = {
                "vlc_left": "sensor", 
                "vlc_right": "sensor",
                "depth": "sensor", 
                "rgb": "sensor",
                "world": "world",
                "body": "body"}
        self.calib_info = {
                "vlc_left": s1,
                "vlc_right": s2,
                "depth": s3,
                "rgb": SensorCalibration(rgb_intrinsics, rgb_extrinsics)
        }
    def update_pv_calibration(self, intrinsics, extrinsics):
        self.calib_info['rgb'] = SensorCalibration(intrinsics,extrinsics)

    def compute_transform(self, f1, f2, pose1, pose2=None):
        '''
        Description:
        ------------
        * pose2 is only used when computing transforms between sensors
        '''
        if self.valid_frames[f1] == "sensor" and self.valid_frames[f2] == "sensor":
            assert pose2 is not None 
            return self.sensor2sensor(f1, f2, pose1, pose2)
        elif self.valid_frames[f1] == "world" and self.valid_frames[f2] == "sensor":
            return self.world2sensor(f2, pose1)
        elif self.valid_frames[f1] == "sensor" and self.valid_frames[f2] == "world":
            return self.sensor2world(f1, pose1) 
        else:
            raise Exception("You naughty coder, you set the wrong combination")

    def sensor2sensor(self, f1, f2, pose1, pose2):
        return self.world2sensor(f2, pose2) @ self.sensor2world(f1, pose1)

    def sensor2world(self, f1, pose1):
        assert self.valid_frames[f1] == "sensor"
        sensor_calib = self.calib_info[f1]
        return sensor_calib.sensor2world(pose1)

    def world2sensor(self, f1, pose1):
        assert self.valid_frames[f1] == "sensor"
        sensor_calib = self.calib_info[f1]
        return sensor_calib.world2sensor(pose1)
