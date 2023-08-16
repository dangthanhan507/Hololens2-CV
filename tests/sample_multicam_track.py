import sys

ROOT_PATH = "../Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

import time

import cv2
import numpy as np
import open3d as o3d
from pynput import keyboard

import cv_utils
from detector import YoloSegment, preprocess_bbox_IOU, IOU_3D
from hl2ss_read import Hl2ssOfflineStreamer
from hl2ss_utils import Hl2ssDepthProcessor
from hl_sensorstack import HololensSensorStack, KinematicChain
from hl2ss_stereo import Hl2ssStereo
from multi_object_tracker import MultiObjectTracker

enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


if __name__ == '__main__':
    print('Starting up Player')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    player = Hl2ssOfflineStreamer('./offline_folder/offline_script5',opts={"vlc_front": True})
    player.open()

    print('Starting detector')
    detector = YoloSegment("yolov8n-seg.pt")
    kin_chain = KinematicChain(player.pv_intrinsics, player.pv_extrinsics)

    sensor_stack = HololensSensorStack()
    depth_processor = Hl2ssDepthProcessor(sensor_stack.calib_lt)

    stereo = Hl2ssStereo()
    tracker = MultiObjectTracker()
 
    while enable:
        data = player.getData()
        if data is None:
            print('Skip')
            continue

        kin_chain.update_pv_calibration(data.color_intrinsics.T, data.color_extrinsics.T)
        sensor_stack.update_pv_calibration(data.color_intrinsics.T, data.color_extrinsics.T)
        data_pv = data.data_pv
        data_lt = data.data_lt
        data_lf = data.data_lf
        data_rf = data.data_rf

        #get rgbd
        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        pts3d_image = cv_utils.rgbd_getpoints_imshape(depth, data.color_intrinsics[:3,:3].T)

        #get grayscale data
        lf,rf = stereo.stereo_pipeline(data_lf.payload, data_rf.payload)
        real_lf = np.dstack((lf,lf,lf))
        real_rf = np.dstack((rf,rf,rf))

        lf, lf_depth = depth_processor.create_rgbd(data_lt, data_lf, sensor_stack.calib_lf.intrinsics,sensor_stack.calib_lf.extrinsics,sensor='lf')
        lf_pts3d_image = cv_utils.rgbd_getpoints_imshape(lf_depth, sensor_stack.calib_lf.intrinsics[:3,:3].T)

        world2vlc_lf_transform = kin_chain.compute_transform("world", "vlc_left", data_lf.pose.T)
        world2rgb_transform = kin_chain.compute_transform("world", "rgb", data_pv.pose.T)
        #detect

        


        # RGB DETECTION FIRST
        bboxes = []
        masks, boxes = detector.eval(rgb,filter_cls=['cup'])
        boxes_mask = preprocess_bbox_IOU(boxes)
        for n in range(len(masks)):
            if boxes_mask[n]:

                mask = masks[n]
                mask = cv2.resize(mask,pts3d_image.shape[:2][::-1],interpolation=cv2.INTER_AREA)
                pts3d_mask = pts3d_image

                pts3d = pts3d_mask.reshape(3,-1)
                pts3d = pts3d[:,mask.flatten() > 0]
                pts3d = pts3d[:,pts3d[2,:] > 0]
                pts3d = np.vstack((pts3d, np.ones((1,pts3d.shape[1]))))
                pts_3d = (data_pv.pose.T @ np.linalg.inv(data.color_extrinsics.T) @ pts3d)[:3,:]

                bbox3d = cv_utils.bbox_3d_from_pcd(pts_3d,name='bbox')
                bboxes.append(bbox3d)

                rgb = boxes[n].drawBox(rgb)

        
        # VLC detection second
        masks, vlc_boxes = detector.eval(real_lf,filter_cls=['cup'])
        for n in range(len(vlc_boxes)):
            mask = masks[n]
            mask = mask[:,::-1].T
            mask = cv2.resize(mask,lf_pts3d_image.shape[:2][::-1],interpolation=cv2.INTER_AREA)
            pts3d_mask = lf_pts3d_image

            pts3d = pts3d_mask.reshape(3,-1)
            pts3d = pts3d[:,mask.flatten() > 0]
            pts3d = pts3d[:,pts3d[2,:] > 0]
            pts3d = np.vstack((pts3d, np.ones((1,pts3d.shape[1]))))
            pts_3d = (data_lf.pose.T @ np.linalg.inv(sensor_stack.calib_lf.extrinsics.T) @ pts3d)[:3,:] # world coords

            real_lf = vlc_boxes[n].drawBox(real_lf)

            rgb_pts2d = sensor_stack.project_onto_sensor(pts_3d, world2rgb_transform, 'rgb')
            rgb_bbox2d = cv_utils.create_bbox(rgb_pts2d, "cup")

            bbox3d = cv_utils.bbox_3d_from_pcd(pts_3d,name='bbox')


            skip = False
            for other_bbox3d in bboxes:
                iou = IOU_3D(bbox3d, other_bbox3d)
                print('wtf: ', iou)
                if iou > 0.4: # too similar!
                    skip = True
                    break
            if skip:
                continue


            bbox3d = cv_utils.bbox_3d_from_pcd(pts_3d,name='bbox')
            bboxes.append(bbox3d)
            

        tracker.track_boxes(bboxes)
        bboxes3d = tracker.getBBoxes()

        cv2.imshow('PV',rgb)
        cv2.imshow('LF', real_lf)
        cv2.waitKey(1)
        time.sleep(1)
    player.close()
    listener.join()
    print('Finishing up Player')