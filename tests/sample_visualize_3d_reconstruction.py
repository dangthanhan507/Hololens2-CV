import sys

ROOT_PATH = "../Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

import time

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import Visualizer
from pynput import keyboard

import cv_utils
from detector import BBox, YoloSegment, preprocess_bbox_IOU
from hl2ss_read import Hl2ssOfflineStreamer
from hl2ss_stereo import Hl2ssStereo
from hl2ss_utils import Hl2ssDepthProcessor
from hl_sensorstack import HololensSensorStack, KinematicChain
from multi_object_tracker import MultiObjectTracker

enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def create_bbox(pts2, class_name):
    xLT = np.min(pts2[0,:])
    yLT = np.min(pts2[1,:])
    xBR = np.max(pts2[0,:])
    yBR = np.max(pts2[1,:])

    bbox2d = BBox(xLT, yLT, xBR, yBR, class_name)
    return bbox2d

def create_transform_matrix_from_z(z):
    """ Return transform 4x4 transformation matrix given a Z value """
    result = np.identity(4)
    result[2,3] = z # Change the z
    
    return result

def dummy_test():
    # Create Open3d visualization window
    vis = Visualizer()
    vis.create_window()

    # create sphere geometry
    sphere1 = o3d.geometry.TriangleMesh.create_sphere()
    vis.add_geometry(sphere1)

    # create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis.add_geometry(coordinate_frame)

    prev_tf = None
    for curr_z in np.arange(0.5, 15.0, 0.005):
        print(curr_z)
        # return sphere1 to original position (0,0,0)
        if prev_tf is not None:
            sphere1.transform(np.linalg.inv(prev_tf))

        # transform bazed on curr_z tf
        curr_tf = create_transform_matrix_from_z(curr_z)
        sphere1.transform(curr_tf)

        prev_tf = curr_tf

        vis.update_geometry(sphere1)
        vis.poll_events()
        vis.update_renderer()

if __name__ == '__main__':
    print('Starting up Player')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    player = Hl2ssOfflineStreamer('./offline_script5',opts={"vlc_front": True})
    player.open()

    detector = YoloSegment("yolov8n-seg.pt")
    tracker = MultiObjectTracker()
    kin_chain = KinematicChain(player.pv_intrinsics, player.pv_extrinsics) 

    sensor_stack = HololensSensorStack()
    depth_processor = Hl2ssDepthProcessor(sensor_stack.calib_lt)
    stereo = Hl2ssStereo()

    vis = Visualizer()
    is_vis_window_created = False

    while enable:
        data = player.getData()
        if data is None:
            continue

        kin_chain.update_pv_calibration(data.color_intrinsics.T, data.color_extrinsics.T)
        data_pv = data.data_pv
        data_lt = data.data_lt

        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        real_depth_img = depth_processor.undistort(depth_processor.get_depthimage(data_lt))
        real_depth_img = real_depth_img[:,:,np.newaxis].astype(float)
        real_depth_img /= real_depth_img.max() 
        pts3d_image = cv_utils.rgbd_getpoints_imshape(depth, data.color_intrinsics[:3,:3].T)
        masks, boxes = detector.eval(rgb, filter_cls=["cup", "bottle"])
        da_cool_kids = preprocess_bbox_IOU(boxes)

        rgb_pose = data_pv.pose.T
        depth_pose = data_lt.pose.T
        for n in range(len(da_cool_kids)):
            if not da_cool_kids[n]:
                continue
            mask = masks[n]
            mask = cv2.resize(mask,pts3d_image.shape[:2][::-1],interpolation=cv2.INTER_AREA)
            pts3d_mask = pts3d_image

            pts3d = pts3d_mask.reshape(3,-1)
            pts3d = pts3d[:,mask.flatten() > 0]
            pts3d = pts3d[:,pts3d[2,:] > 0]
            pts3d = np.vstack((pts3d, np.ones((1,pts3d.shape[1]))))
            pts_3d = (data_pv.pose.T @ np.linalg.inv(data.color_extrinsics.T) @ pts3d)[:3,:]

            if not is_vis_window_created:
                is_vis_window_created = True
                vis.create_window()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_3d.T)
            normals = np.zeros_like(np.asarray(pcd.points))
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            # Alpha Mesh
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.03)
            mesh.compute_vertex_normals()

            vis.add_geometry(mesh)
            vis.update_geometry(mesh)
            vis.poll_events()
            vis.update_renderer()

            rgb = boxes[n].drawBox(rgb)
            
        cv2.imshow('D',real_depth_img)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(1)
    vis.destroy_window()
    player.close()
    listener.join()
    print('Finishing up Player')
