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
from render_lib import DetBox, setMultiObjectPose
import traceback

enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def main(streamer, render):
    depth_processor = Hl2ssDepthProcessor(streamer.calib_lt)
    detector = YoloSegment("yolov8n-seg.pt")
    tracker = MultiObjectTracker()


    while enable:
        streamer.waitReady()
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
            print(pts_3d.shape)
            bbox3d = cv_utils.bbox_3d_from_pcd(pts_3d,name='bbox')
            bboxes.append(bbox3d)

        print("Num of bboxes:", len(bboxes))
        tracker.track_boxes(bboxes)

        bboxes3d = tracker.getBBoxes()

        pv_pose = data.data_pv.pose.T
        render.clear()
        for bbox3d in bboxes3d:
            center_pt = bbox3d.getCenter()
            pose = cv_utils.calc_pose_xz(pv_pose, center_pt)

            renderbbox = DetBox(bbox3d,thickness=0.01)

            bbox3d_objs = renderbbox.create_render()
            # bbox3d_objs = renderbbox.setWindowPose(pose)
            bbox3d_objs_ids = render.addPrimObjects(bbox3d_objs)

        cv2.imshow('D',depth)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(1)


if __name__ == '__main__':
    print('Starting up Server')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    streamer = Hl2ssStreamWrapper()
    render = Hl2ssRender()
    streamer.start()
    render.start()
    
    try:
        main(streamer,render)
    except:
        type, value, _ = sys.exc_info()
        print(value)
        traceback.print_exc()


    streamer.stop()
    listener.join()

    print('Finishing up Server')
