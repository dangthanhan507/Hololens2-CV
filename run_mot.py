import sys

from detector import YoloSegment

ROOT_PATH = "../Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

import time

import cv2
import numpy as np
import open3d as o3d
from pynput import keyboard

import cv_utils
from hl2ss_map import Hl2ssMapping
from hl2ss_read import Hl2ssOfflineStreamer
from hl2ss_utils import Hl2ssDepthProcessor
from hl_sensorstack import HololensSensorStack
from multi_object_tracker import MultiObjectTracker
from render_lib import RenderBBox

enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

if __name__ == '__main__':
    print('Starting up Player')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    player = Hl2ssOfflineStreamer('./offline_script0',{})
    player.open()

    sensor_stack = HololensSensorStack()
    depth_processor = Hl2ssDepthProcessor(sensor_stack.calib_lt)

    detector = YoloSegment("yolov8n-seg.pt")
    mot = MultiObjectTracker()
 
    while enable:
        data = player.getData()
        if data is None:
            continue

        data_pv = data.data_pv
        data_lt = data.data_lt
        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        pts3d_image = cv_utils.rgbd_getpoints_imshape(depth, data.color_intrinsics[:3,:3].T)
        masks, boxes = detector.eval(rgb, filter_cls=["toilet"])

        depth_mask = (np.zeros(pts3d_image.shape[:2]) != 0)
        bboxes = []
        for n in range(len(masks)):
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

        print("Num of bboxes:", len(bboxes))
        mot.track_boxes(bboxes)

        cv2.imshow('D',depth)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(0.02)

    player.close()
    listener.join()
    print('Finishing up Player')
