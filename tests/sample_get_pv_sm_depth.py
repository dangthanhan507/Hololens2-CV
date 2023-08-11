import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData, PV_WIDTH, PV_HEIGHT
from hl2ss_utils import Hl2ssDepthProcessor
from hl2ss_map import Hl2ssMapping
from hl_sensorstack import KinematicChain
from pynput import keyboard
import cv2
import numpy as np
import hl2ss_utils
import hl2ss_3dcv


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
    mapping = Hl2ssMapping()
    mapping.observe_map()

    depth_processor = Hl2ssDepthProcessor(streamer.calib_lt)
    streamer.start()

    kin_chain = KinematicChain(streamer.pv_intrinsics, streamer.pv_extrinsics)
    

    
    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        data_pv = data.data_pv
        data_lt = data.data_lt

        pv_rays = hl2ss_utils.create_rays(data.color_intrinsics, PV_WIDTH, PV_HEIGHT)

        kin_chain.update_pv_calibration(data.color_intrinsics.T, data.color_extrinsics.T)

        cam2world = kin_chain.compute_transform('rgb','world', data_pv.pose.T)
        # cam2world = data_pv.pose.T @ np.linalg.inv(data.color_extrinsics.T)
        cam2world = cam2world.T


        pv_world_origins = np.tile(cam2world[3, :3], (PV_HEIGHT, PV_WIDTH, 1))

        pv_world_directions = pv_rays @ cam2world[:3,:3]
        pv_world_rays = np.dstack((pv_world_origins,pv_world_directions))

        pv_depth = mapping.rays2depth(pv_world_rays)
        pv_depth[np.isinf(pv_depth)] = 0

        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)

        diff_depth = np.zeros(depth.shape)
        diff_depth[depth>0] = np.abs(pv_depth[depth>0] - depth[depth>0] )

        cv2.imshow('PV_D',pv_depth / pv_depth.max())
        cv2.imshow('D',depth / depth.max())
        cv2.imshow('Difference', diff_depth / diff_depth.max())
        cv2.waitKey(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')