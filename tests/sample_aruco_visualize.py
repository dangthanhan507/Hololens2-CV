import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_render import Hl2ssRender
from hl2ss_utils import Hl2ssDepthProcessor
from pynput import keyboard
import cv2
import numpy as np
import time
from cv_utils import pts2d_to_pts3d

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


    #aruco stuff
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    parameters = cv2.aruco.DetectorParameters()
    detect_aruco = cv2.aruco.ArucoDetector(dictionary, parameters)

    rotation = [0, 0, 0, 1]
    scale = [0.01,0.01,0.01]
    rgba = [1,0,0,1]
    while enable:
        render.clear()
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        data_pv = data.data_pv
        data_lt = data.data_lt
        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)

        #show detection
        corners,idx,rejected = detect_aruco.detectMarkers(rgb)
        for corner in corners:
            cr = corner.squeeze().T.astype(np.int32)
            pts3d = pts2d_to_pts3d(cr, depth, data.color_intrinsics[:3,:3].T)
            _,N = pts3d.shape
            pts3d_h = np.vstack((pts3d,np.ones((1,N))))
            pts3d = (data_pv.pose.T @ np.linalg.inv(data.color_extrinsics).T @  pts3d_h )[:3,:]           

            # print(pts3d)
            for n in range(N):
                pos = pts3d[:,n].tolist()
                render.addPrimObject("sphere", pos, rotation, scale, rgba)

        #TODO: fix reprojection depth having holes in the depth
        depth = depth / depth.max() * 255
        depth = np.dstack((depth,depth,depth))
        depth = cv2.aruco.drawDetectedMarkers(np.ascontiguousarray(depth, dtype=np.uint8), corners)
        rgb = cv2.aruco.drawDetectedMarkers(np.ascontiguousarray(rgb, dtype=np.uint8), corners)

        cv2.imshow('RGB',rgb)
        cv2.imshow('D',depth)
        cv2.waitKey(1)
        time.sleep(5)
        

    


    print('Finishing up Server')
    render.clear()
    render.stop()
    streamer.stop()
    listener.join()
    