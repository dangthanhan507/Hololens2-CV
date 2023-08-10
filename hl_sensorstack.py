#------------------------------------------------------------------------------
# hl_sensorstack.py
# 
# This file should have code representing the code to move between sensors
# in the entire hololens sensor stack. There's a lot of projection/unprojection
# needed to be done. This class should have all of that processing done.
#------------------------------------------------------------------------------
from hl2ss_stream import HOST_IP, ROOT_PATH
import os
import sys
sys.path.append(os.path.join(ROOT_PATH,'hl2ss','viewer'))
import numpy as np

CALIB_PATH = os.path.join(ROOT_PATH,'hl2ss','calibration')

import hl2ss
import hl2ss_3dcv
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
    def __init__(self):
        self.calib_lt = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, os.path.join(CALIB_PATH,"rm_depth_longthrow"))
        uv2xy = hl2ss_3dcv.compute_uv2xy(self.calib_lt.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        self.xy1, self.lt_scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, self.calib_lt.scale)

        self.calib_lf = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_VLC_LEFTFRONT, os.path.join(CALIB_PATH,"rm_vlc_leftfront"))
        self.calib_rf = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, os.path.join(CALIB_PATH,"rm_vlc_rightfront"))

        
    def update_pv_calibration(self, pv_intrinsics, pv_extrinsics):
        pass
    def undistort(self, undistort_map, data):
        pass


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
        return np.linalg.inv(self.extrisnics)
    def body2sensor(self):
        return self.extrinsics
    
    def body2world(self, pose):
        return pose
    def world2body(self, pose):
        return np.linalg.inv(pose)
    
    def world2sensor(self, pose):
        return self.world2body(pose) @ self.body2sensor()
    def sensor2world(self, pose):
        return self.body2world(pose) @ self.sensor2body()
class KinematicChain:
    def __init__(self):
        pass