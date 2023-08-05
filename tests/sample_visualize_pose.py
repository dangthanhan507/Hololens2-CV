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

    rotation = [0, 0, 0, 1]
    scale = np.array([0.1,0.1,0.1])*1
    rgba = [1,0,0,1]
    while enable:
        render.clear()
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        data_pv = data.data_pv
        
        render.addPrimObject('sphere', [0,0,0], rotation, scale.tolist(), [1,1,1,1])

        pos = data_pv.pose.T[:3,-1].flatten()
        pos[2] *= -1
        render.addPrimObject("sphere", pos, rotation, scale.tolist(), [0,1,0,1])
        print(data_pv.pose.T[:3,-1].flatten())
        time.sleep(5)
        

    


    print('Finishing up Server')
    render.clear()
    render.stop()
    streamer.stop()
    listener.join()
    