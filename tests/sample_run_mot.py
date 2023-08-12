import sys

ROOT_PATH = "../Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

import time

import cv2
import numpy as np
import open3d as o3d
from pynput import keyboard

import cv_utils
from detector import BBox, YoloSegment, preprocess_bbox_IOU
from hl2ss_map import Hl2ssMapping
from hl2ss_read import Hl2ssOfflineStreamer
from hl2ss_stereo import Hl2ssStereo
from hl2ss_utils import Hl2ssDepthProcessor
from hl_sensorstack import HololensSensorStack, KinematicChain
from multi_object_tracker import MultiObjectTracker
from render_lib import RenderBBox

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

if __name__ == '__main__':
    print('Starting up Player')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    player = Hl2ssOfflineStreamer('./offline_script0',opts={"vlc_front": True})
    player.open()

    detector = YoloSegment("yolov8n-seg.pt")
    tracker = MultiObjectTracker()
    kin_chain = KinematicChain(player.pv_intrinsics, player.pv_extrinsics) 

    sensor_stack = HololensSensorStack(kin_chain)
    depth_processor = Hl2ssDepthProcessor(sensor_stack.calib_lt)
    stereo = Hl2ssStereo()
 
    while enable:
        data = player.getData()
        if data is None:
            continue

        kin_chain.update_pv_calibration(data.color_intrinsics.T, data.color_extrinsics.T)
        data_pv = data.data_pv
        data_lt = data.data_lt
        data_lf = data.data_lf
        data_rf = data.data_rf

        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        real_depth_img = depth_processor.undistort(depth_processor.get_depthimage(data_lt))
        real_depth_img = real_depth_img[:,:,np.newaxis].astype(float)
        real_depth_img /= real_depth_img.max() 
        pts3d_image = cv_utils.rgbd_getpoints_imshape(depth, data.color_intrinsics[:3,:3].T)
        masks, boxes = detector.eval(rgb, filter_cls=["toilet"])
        da_cool_kids = preprocess_bbox_IOU(boxes)

        lf = data.data_lf.payload
        rf = data.data_rf.payload
        real_lf = lf[:,:,np.newaxis]
        real_rf = rf[:,:,np.newaxis]

        bboxes = []
        rgb_pose = data_pv.pose.T
        depth_pose = data_lt.pose.T
        world2depth_transform = kin_chain.compute_transform("world", "depth", depth_pose)
        world2vlc_lf_transform = kin_chain.compute_transform("world", "vlc_left", data_lf.pose.T)
        world2vlc_rf_transform = kin_chain.compute_transform("world", "vlc_right", data_rf.pose.T)
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
            
            #get info for 3d bboxs
            bbox3d = cv_utils.bbox_3d_from_pcd(pts_3d,name='bbox')
            bboxes.append(bbox3d)

            rgb = boxes[n].drawBox(rgb)

            depth_pts2d = sensor_stack.project_onto_depth_frame(pts_3d, world2depth_transform)
            bbox2d = create_bbox(depth_pts2d, "toilet")
            bbox2d.drawBox(real_depth_img)

            vlc_lf_pts2d = sensor_stack.project_onto_vlc_sensor(pts_3d, world2vlc_lf_transform, "left")
            vlc_lf_bbox2d = create_bbox(vlc_lf_pts2d, "toilet")
            vlc_lf_bbox2d.drawBox(real_lf)

            vlc_rf_pts2d = sensor_stack.project_onto_vlc_sensor(pts_3d, world2vlc_rf_transform, "right")
            vlc_rf_bbox2d = create_bbox(vlc_rf_pts2d, "toilet")
            vlc_rf_bbox2d.drawBox(real_rf)

        tracker.track_boxes(bboxes)
        bbox3d_pts = tracker.get_bbox_3d_pts()


        cv2.imshow('LF', real_lf)
        cv2.imshow('RF', real_rf)
        cv2.imshow('D',real_depth_img)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(1)
    player.close()
    listener.join()
    print('Finishing up Player')
