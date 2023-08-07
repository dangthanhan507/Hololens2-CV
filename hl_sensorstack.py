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
    def undistort(self, undistort_map, data):
        pass