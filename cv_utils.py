#------------------------------------------------------------------------------
# cv_utils.py
# 
# This is code to process computer vision output and utilize for purposes
# of the hololens. It should be filled with miscellaneous functions
# thar are useful for the computation of more specialized algorithms that
# belong in other files.
#------------------------------------------------------------------------------

import numpy as np
from detector import BBox, BBox3D
from scipy.spatial.transform import Rotation


def create_bbox(pts2, class_name):
    xLT = np.min(pts2[0,:])
    yLT = np.min(pts2[1,:])
    xBR = np.max(pts2[0,:])
    yBR = np.max(pts2[1,:])

    bbox2d = BBox(xLT, yLT, xBR, yBR, class_name)
    return bbox2d

def calc_pose_xz(pose, obj_pos):
    '''
        Calculate your xz angle from an object given your pose
        and object position

        obj_pos (3,1)
        pose (4,4)

    '''
    x = pose[0,-1]
    z = pose[2,-1]

    xt = obj_pos[0,0]
    zt = obj_pos[2,0]

    #calculate pos relative to obj
    x = x - xt
    z = z - zt

    angle = np.arctan2(z,x)
    rot_mat = Rotation.from_rotvec(np.array([0,-angle - np.pi/2,0])).as_matrix()

    pose = np.eye(4)
    pose[:3,:3] = rot_mat
    return pose



def rgbd_getpoints_imshape(depth, pv_intrinsics):
    m,n = depth.shape
    pts3d = rgbd_getpoints(depth, pv_intrinsics)
    return pts3d.reshape(m,n,3)

def rgbd_getpoints(depth, pv_intrinsics):
    '''
    Description:
    ------------
        Takes an image with a corresponding intrinsic and 
    Parameters:
    -----------
        depth: (M,N) image representing depth in meters for each pixel
        pv_intrinsics: (3x3) matrix representing pinhole intrinsic matrix

    Returns:
    --------
        point_clouds: (3,N) matrix representing point clouds with each point
                      represented as a column.
    '''
    height,width = depth.shape
    yy,xx = np.mgrid[:height,:width]
    
    yy = yy.flatten()
    xx = xx.flatten()
    depth_points = depth.flatten()

    #2d homogenous points in image coords
    pts2d_h = np.vstack((xx,yy,np.ones(xx.shape[0])))

    '''
    NOTE:
    ------
    
    camera intrinsic projects 3d point.
    then normalize out z dimension to get the image points
    shown below....

    |fx  0 cx|   |X|   |x|    |x_i|
    | 0 fy cy| * |Y| = |y| => |y_i|
    | 0  0  1|   |Z|   |z|    |  1|

    We intend to do the opposite
    '''
    
    #unnormalize z point
    pts3d_h = pts2d_h * depth_points
    #undo intrinsic
    pts3d = np.linalg.inv(pv_intrinsics) @ pts3d_h
    return pts3d

def pts2d_to_pts3d(pts2d, depth, pv_intrinsics):
    
    pts2d_h = np.vstack((pts2d,np.ones(pts2d.shape[1])))
    depth_points = depth[pts2d[1,:], pts2d[0,:]]
    pts3d = pts2d_h * depth_points
    pts3d = np.linalg.inv(pv_intrinsics) @ pts2d_h

    #filter only for valid 3d points (z != 0)
    pts3d = pts3d[:,pts3d[2,:] > 0]
    return pts3d
        

    

def bbox_getdepth(depth, bbox, pv_intrinsics):

    height,width = depth.shape
    yy,xx = np.mgrid[:height,:width]

    x0,y0 = bbox.getTL() #top left 
    x1,y1 = bbox.getBR() #bot right
    x0,y0 = int(x0), int(y0)
    x1,y1 = int(x1), int(y1)

    mask = (xx >= x0) | (xx <= x1) | (yy >= y0) | (yy <= y1)
    xx = xx[mask]
    yy = yy[mask]
    depth_o = depth[mask]

    yy = yy.flatten()
    xx = xx.flatten()
    depth_points = depth_o.flatten()
    pts2d_h = np.vstack((xx,yy,np.ones(xx.shape[0])))
    pts3d_h = pts2d_h * depth_points
    pts3d = np.linalg.inv(pv_intrinsics) @ pts3d_h
    pts3d = pts3d[:,pts3d[2,:] > 0]
    return pts3d

def seg_getdepth(depth, seg, pv_intrinsics):

    height,width = depth.shape
    yy,xx = np.mgrid[:height,:width]

    mask = (seg==1)
    depth_o = depth[mask]
    xx = xx[mask]
    yy = yy[mask]

    yy = yy.flatten()
    xx = xx.flatten()
    depth_points = depth_o.flatten()
    pts2d_h = np.vstack((xx,yy,np.ones(xx.shape[0])))
    pts3d_h = pts2d_h * depth_points
    pts3d = np.linalg.inv(pv_intrinsics) @ pts3d_h
    return pts3d


def bbox_3d_from_pcd(pts3d, name):
    '''
    create 3d bounding box from set of 3d point clouds.
    axis aligned 3d bounding box

    pts3d: (3,N)
    '''
    min_box = pts3d.min(axis=1).flatten().tolist()
    max_box = pts3d.max(axis=1).flatten().tolist()

    return BBox3D( *(min_box + max_box + [name]) )