#------------------------------------------------------------------------------
# hl2ss_stereo.py
# 
# Hl2ss Stereo Module which will go from hl2ss data to stereo rectified.
# We can then use this to get point clouds using stereo vision
#------------------------------------------------------------------------------
from hl2ss_stream import HOST_IP, ROOT_PATH
import os
import sys
sys.path.append(os.path.join(ROOT_PATH,'hl2ss','viewer'))
import numpy as np

CALIB_PATH = os.path.join(ROOT_PATH,'hl2ss','calibration')

import hl2ss
import hl2ss_3dcv
import cv2

class Hl2ssStereo:
    def __init__(self):
        self.calib_lf = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_VLC_LEFTFRONT, os.path.join(CALIB_PATH,"rm_vlc_leftfront"))
        self.calib_rf = hl2ss_3dcv._load_calibration_rm(hl2ss.StreamPort.RM_VLC_RIGHTFRONT, os.path.join(CALIB_PATH,"rm_vlc_rightfront"))

        self.rotation_lf = hl2ss_3dcv.rm_vlc_get_rotation(hl2ss.StreamPort.RM_VLC_LEFTFRONT)
        self.rotation_rf = hl2ss_3dcv.rm_vlc_get_rotation(hl2ss.StreamPort.RM_VLC_RIGHTFRONT)

        K1, Rt1 = hl2ss_3dcv.rm_vlc_rotate_calibration(self.calib_lf.intrinsics, self.calib_lf.extrinsics, self.rotation_lf)
        K2, Rt2 = hl2ss_3dcv.rm_vlc_rotate_calibration(self.calib_rf.intrinsics, self.calib_rf.extrinsics, self.rotation_rf)

        self.stereo_calibration = hl2ss_3dcv.rm_vlc_stereo_calibrate(K1,K2,Rt1,Rt2)
        self.stereo_rectification = hl2ss_3dcv.rm_vlc_stereo_rectify(K1, K2, self.stereo_calibration.R, self.stereo_calibration.t, hl2ss.Parameters_RM_VLC.SHAPE)

        #in stereo calculations, this is the "baseline" after we set it to be rectified
        #whatever dimension this is in, will be the dimension of the point clouds
        #NOTE: assumed dimension is in meters
        self.stereo_baseline = np.linalg.norm(self.stereo_calibration.t)

    def undistort_stereo(self, image_lf, image_rf):
        '''
            image_lf: (MxN)
            image_rf: (MxN)
        '''
        image_lf = hl2ss_3dcv.rm_depth_undistort(image_lf, self.calib_lf.undistort_map)
        image_rf = hl2ss_3dcv.rm_depth_undistort(image_rf, self.calib_rf.undistort_map)

        return image_lf, image_rf
    def fix_hl2ss_rot(self, image_lf, image_rf):
        image_lf = hl2ss_3dcv.rm_vlc_rotate_image(image_lf, self.rotation_lf)
        image_rf = hl2ss_3dcv.rm_vlc_rotate_image(image_rf, self.rotation_rf)
        return image_lf, image_rf
    
    def rectify_stereo(self, image_lf, image_rf):
        rectified_lf = cv2.remap(image_lf, self.stereo_rectification.map1[:, :, 0], self.stereo_rectification.map1[:, :, 1], cv2.INTER_LINEAR)
        rectified_rf = cv2.remap(image_rf, self.stereo_rectification.map2[:, :, 0], self.stereo_rectification.map2[:, :, 1], cv2.INTER_LINEAR)
        return rectified_lf, rectified_rf
    
    def stereo_pipeline(self, image_lf, image_rf):
        image_lf, image_rf = self.undistort_stereo(image_lf, image_rf)
        image_lf, image_rf = self.fix_hl2ss_rot(image_lf, image_rf)
        image_lf, image_rf = self.rectify_stereo(image_lf, image_rf)
        return image_lf, image_rf
    
    def make_stereo_display(self, image_lf, image_rf):
        image_l = hl2ss_3dcv.rm_vlc_to_rgb(image_lf)
        image_r = hl2ss_3dcv.rm_vlc_to_rgb(image_rf)

        image = np.hstack((image_l, image_r))
        return image
    