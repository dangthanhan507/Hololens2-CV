import sys

ROOT_PATH = "../Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

import time

import cv2
import numpy as np
from pynput import keyboard

import cv_utils
from detector import BBox, YoloDetector, YoloSegment
from hl2ss_render import Hl2ssRender, RenderObject
from hl2ss_stream import Hl2ssData, Hl2ssStreamWrapper
from hl2ss_utils import Hl2ssDepthProcessor
from multi_object_tracker import MultiObjectTracker

enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


if __name__ == '__main__':
    print('Starting up Server')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    streamer = Hl2ssStreamWrapper()
    render = Hl2ssRender()
    depth_processor = Hl2ssDepthProcessor(streamer.calib_lt)
    detector = YoloSegment("yolov8n-seg.pt")
    tracker = MultiObjectTracker()
    streamer.start()
    render.start()


    rotation = [0, 0, 0, 1]
    scale = np.array([0.1,0.1,0.1])*0.5
    rgba = [1,1,0,1]
    detector = YoloDetector("yolov8n.pt")

    while enable:
        streamer.waitReady()
        render.clear()
        data = streamer.getData()
        if data is None:
            print('Skipped')
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
        tracker.track_boxes(bboxes)

        cv2.imshow('D',depth)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')
