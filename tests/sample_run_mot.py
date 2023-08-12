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

def draw_bbox2d(bbox_pts2d, frame):
    for i in range(0, bbox_pts2d.shape[1], 2):
        pts2d = bbox_pts2d[:,2*i: 2*i+2].flatten(order='F').tolist()
        bbox = BBox(*(pts2d + ["bbox"]))
        bbox.drawBox(frame)

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
        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        real_depth_img = depth_processor.undistort(depth_processor.get_depthimage(data_lt))
        real_depth_img = real_depth_img[:,:,np.newaxis].astype(float)
        real_depth_img /= real_depth_img.max() 
        pts3d_image = cv_utils.rgbd_getpoints_imshape(depth, data.color_intrinsics[:3,:3].T)
        masks, boxes = detector.eval(rgb, filter_cls=["toilet"])
        da_cool_kids = preprocess_bbox_IOU(boxes)

        bboxes = []
        rgb_pose = data_pv.pose.T
        depth_pose = data_lt.pose.T
        rgb2depth_transform = kin_chain.compute_transform("world", "depth", depth_pose)
        # depth2vlc_lf_transform = kin_chain.compute_transform()
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

            depth_bbox_pts = sensor_stack.project_onto_depth_frame(pts_3d, rgb2depth_transform)
            depth_bbox_pts2d, depth_bbox_pts3d = depth_bbox_pts
            xLT = np.min(depth_bbox_pts2d[0,:])
            yLT = np.min(depth_bbox_pts2d[1,:])
            xBR = np.max(depth_bbox_pts2d[0,:])
            yBR = np.max(depth_bbox_pts2d[1,:])

            bbox2d = BBox(xLT, yLT, xBR, yBR, "toilet")
            bbox2d.drawBox(real_depth_img)

            vlc_lf_pts = depth_processor.project_onto_vlc_sensor(depth_bbox_pts3d, data_lt.pose.T, data.data_lf.pose.T, "left")
            vlc_lf_bbox_pts2d, vlc_lf_bbox_pts3d = vlc_lf_pts
            draw_bbox2d(vlc_lf_bbox_pts2d, lf)

            vlc_rf_pts = depth_processor.project_onto_vlc_sensor(depth_bbox_pts3d, data_lt.pose.T, data.data_lf.pose.T, "right")
            vlc_rf_bbox_pts2d, vlc_rf_bbox_pts3d = vlc_rf_pts
            draw_bbox2d(vlc_rf_bbox_pts3d, rf)

            cv2.imshow('LF', lf)
            cv2.imshow('RF', rf)
        tracker.track_boxes(bboxes)
        bbox3d_pts = tracker.get_bbox_3d_pts()


        cv2.imshow('D',real_depth_img)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(1)
    player.close()
    listener.join()
    print('Finishing up Player')
