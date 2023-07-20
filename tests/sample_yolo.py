import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from detector import BBox, YoloDetector
from hl2ss_render import Hl2ssRender
from hl2ss_utils import Hl2ssDepthProcessor
from pynput import keyboard
import cv2
import cv_utils
import numpy as np
import time

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
    streamer.start()
    render.start()


    rotation = [0, 0, 0, 1]
    scale = np.array([0.1,0.1,0.1])*0.1
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
        # pts3d_image = cv_utils.rgbd_getpoints_imshape(depth, data.color_intrinsics[:3,:3].T)
        bboxes = detector.eval(rgb)
        for bbox in bboxes:
            rgb = bbox.drawBox(rgb)

            x0,y0 = bbox.getTL()
            x1,y1 = bbox.getBR()
            x0,y0 = int(x0), int(y0)
            x1,y1 = int(x1), int(y1)
            # pts3d_bbox = pts3d_image[y0:y1, x0:x1]
            # pts3d = pts3d_bbox.reshape(3,-1)
            pts3d = cv_utils.bbox_getdepth(depth, bbox, data.color_intrinsics[:3,:3].T)
            pts3d = pts3d[:,pts3d[2,:] != 0]
            pts3d = np.mean(pts3d,axis=1).reshape(3,1)
            pts3d = np.vstack((pts3d, np.ones((1,pts3d.shape[1]))))
            pts_3d = (data_pv.pose.T @ np.linalg.inv(data.color_extrinsics.T) @ pts3d)[:3,:]
            pos = pts_3d.flatten().tolist()
            pos[2] *= -1
            print(bbox.name, pos)

            render.addPrimObject("sphere", pos, rotation, scale.tolist(), rgba)

            cv2.imshow('rgb bbox', rgb[y0:y1,x0:x1])
            cv2.imshow('depth bbox', depth[y0:y1, x0:x1])
            cv2.waitKey(1)




        cv2.imshow('D',depth)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')