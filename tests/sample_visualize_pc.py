import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_render import Hl2ssRender
from hl2ss_utils import Hl2ssDepthProcessor
from pynput import keyboard
import cv2
from cv_utils import rgbd_getpoints
import numpy as np
import time

enable = True
'''
DON'T RUN THIS. TAKES FOREVER TO FINISH RENDERING.
I RAN THIS ALREADY AND IT TOOK 20 minutes to work.

Although the proof of concept is great. Visualized 3d details of my room with rgbd
data that I used.

Everything works as expected!!!

'''
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

    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        data_pv = data.data_pv
        data_lt = data.data_lt

        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)


        # (3xN)
        pts3d = rgbd_getpoints(depth, data.color_intrinsics[:3,:3].T)
        pts3d = np.vstack((pts3d, np.ones((1,pts3d.shape[1]))))

        pts_3d = (data_pv.pose.T @ np.linalg.inv(data.color_extrinsics.T) @ pts3d)[:3,:]

        # reverse Z direction for visualizing
        for n in range(pts_3d.shape[1]):
            pos = pts_3d[:3,n].tolist()
            pos[2] *= -1
            render.addPrimObject("sphere", pos, rotation, scale.tolist(), rgba)

        cv2.imshow('D',depth)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')