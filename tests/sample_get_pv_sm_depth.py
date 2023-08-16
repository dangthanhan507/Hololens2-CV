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

        lt_height, lt_width = data_lt.payload.depth.shape[:2]
        lt_rays = hl2ss_utils.create_rays(streamer.calib_lt.intrinsics, lt_width, lt_height)

        kin_chain.update_pv_calibration(data.color_intrinsics.T, data.color_extrinsics.T)

        cam2world = kin_chain.compute_transform('rgb','world', data_pv.pose.T)
        cam2world = cam2world.T
        tof2world = kin_chain.compute_transform('depth', 'world', data_lt.pose.T)
        tof2world = tof2world.T

        pv_world_origins = np.tile(cam2world[3, :3], (PV_HEIGHT, PV_WIDTH, 1))
        pv_world_directions = pv_rays @ cam2world[:3,:3]
        pv_world_rays = np.dstack((pv_world_origins,pv_world_directions))

        pv_depth = mapping.rays2depth(pv_world_rays)
        pv_depth[np.isinf(pv_depth)] = 0

        lt_world_origins = np.tile(tof2world[3, :3], (lt_height, lt_width, 1))
        lt_world_directions = lt_rays @ tof2world[:3,:3]
        lt_world_rays = np.dstack((lt_world_origins,lt_world_directions))

        lt_depth = mapping.rays2depth(lt_world_rays)
        lt_depth[np.isinf(lt_depth)] = 0
        

        # rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        depth = depth_processor.get_depthimage(data_lt)
        depth = depth_processor.undistort(depth)
        depth = depth_processor.normalize(depth).squeeze()

        print(lt_depth.max())
        print(depth.max())
        print()

        boolean_mask = (depth>0) & (lt_depth > 0)
        diff_depth = np.zeros(depth.shape)
        diff_depth[boolean_mask] = np.abs(lt_depth[boolean_mask] - depth[boolean_mask] )

        mask = np.zeros(depth.shape)
        mask[(diff_depth > 0.3)] = 1

        cv2.imshow('PV_D',lt_depth / lt_depth.max())
        cv2.imshow('D',depth / depth.max())
        cv2.imshow('Difference', diff_depth / diff_depth.max())
        cv2.imshow('Mask', mask)
        cv2.waitKey(1)
        time.sleep(0.5)

    streamer.stop()
    listener.join()

    print('Finishing up Server')