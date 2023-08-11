#------------------------------------------------------------------------------
# hl2ss_utils.py
# 
# This file is meant to use hl2ss data to perform calculations that are
# listed under pre-processing for hl2ss. For example, creating rgbd
# is something we perform as a pre-process step after getting our data.
# Ideally, we should never reference hl2ss in the scripts and only on this.
# This would make it easy to work with.
#------------------------------------------------------------------------------

import os
import sys

import hl2ss_stream

sys.path.append(os.path.join(hl2ss_stream.ROOT_PATH,'hl2ss','viewer'))
import hl2ss_3dcv
import numpy as np

import hl2ss
from hl_sensorstack import KinematicChain

# stick kinematic chain in here


class Hl2ssDepthProcessor:
    def __init__(self, calibration):
        self.calibration = calibration
        self.uv2xy = hl2ss_3dcv.compute_uv2xy(calibration.intrinsics, hl2ss.Parameters_RM_DEPTH_LONGTHROW.WIDTH, hl2ss.Parameters_RM_DEPTH_LONGTHROW.HEIGHT)
        self.xy1, self.scale = hl2ss_3dcv.rm_depth_compute_rays(self.uv2xy, calibration.scale)
        self.k_chain = KinematicChain(np.eye(3), np.eye(4)) # DUMMY VALUES FOR NOW
    
    def get_depthimage(self, data_lt):
        return data_lt.payload.depth
    def undistort(self, depth):
        return hl2ss_3dcv.rm_depth_undistort(depth, self.calibration.undistort_map)
    def normalize(self, depth):
        return hl2ss_3dcv.rm_depth_normalize(depth,self.scale)
    def unproject_depthpoints(self, depth):
        return hl2ss_3dcv.rm_depth_to_points(self.xy1, depth)
    def get_worldpoints(self, lt_points, data_lt):
        lt_to_world = hl2ss_3dcv.camera_to_rignode(self.calibration.extrinsics) @ hl2ss_3dcv.reference_to_world(data_lt.pose)
        return hl2ss_3dcv.transform(lt_points, lt_to_world)
    
    def create_rgbd(self, data_lt, data_pv, pv_intrinsics, pv_extrinsics):
        pv_im = get_pv_image(data_pv)
        height,width,_ = pv_im.shape

        depth = self.get_depthimage(data_lt)
        depth = self.undistort(depth)
        depth = self.normalize(depth)
        lt_points = self.unproject_depthpoints(depth)
        world_points = self.get_worldpoints(lt_points, data_lt)

        world_to_pv = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
        unnorm_pts = hl2ss_3dcv.transform(world_points, world_to_pv).reshape(-1,3).T
        norm_pts = unnorm_pts / unnorm_pts[2,:]
        mask = (norm_pts[0,:] < 0) | (norm_pts[0,:] >= width) | (norm_pts[1,:] < 0) | (norm_pts[1,:] >= height)
        unnorm_pts = unnorm_pts[:,~mask]
        norm_pts = norm_pts[:,~mask]

        depth_orig = np.zeros((height,width))
        depth_orig[np.int32(norm_pts[1,:]), np.int32(norm_pts[0,:])] = unnorm_pts[2,:]
        return pv_im, depth_orig

    def project_onto_depth_frame(self, rgb_bbox_pts3):
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
        rgb_pose = self.k_chain.calib_info["rgb"].extrinsics[:,-1].reshape((4,1))
        depth_pose = self.k_chain.calib_info["rgb"].extrinsics[:,-1].reshape((4,1))
        transform = self.k_chain.compute_transform("rgb", "depth", rgb_pose, depth_pose)

        depth_intrinsics = self.k_chain.calib_info["depth"].intrinsics
        fx, fy = depth_intrinsics[0,0], depth_intrinsics[1,1]
        cx, cy = depth_intrinsics[0,2], depth_intrinsics[1,2]
        rgb_bbox_pts3 = np.vstack((rgb_bbox_pts3, np.zeros((rgb_bbox_pts3.shape[1],))))
        rgb_bbox_pts3[-1,-1] = 1
        depth_bbox_pts3 = transform @ rgb_bbox_pts3
        depth_bbox_pts2 = depth_bbox_pts3[0:2,:]/depth_bbox_pts3[2]
        depth_bbox_pts2[0,:] = (fx * depth_bbox_pts2[0,:]) + cx
        depth_bbox_pts2[1,:] = (fy * depth_bbox_pts2[1,:]) + cy
        return depth_bbox_pts2, depth_bbox_pts3

    def project_onto_vlc_lf_sensor(self, depth_bbox_pts3):
        pass
        
    def project_onto_vlc_rf_sensor(self, depth_bbox_pts3):
        pass

def get_pv_image(data_pv):
    return data_pv.payload.image

def create_rays(intrinsics, width, height):
    rays = hl2ss_3dcv.compute_uv2xy(intrinsics,width,height)
    rays = hl2ss_3dcv.to_homogeneous(rays)
    rays = hl2ss_3dcv.to_unit(rays)
    return rays
