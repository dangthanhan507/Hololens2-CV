import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_utils import Hl2ssDepthProcessor
from pynput import keyboard
import cv2

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
    depth_processor = Hl2ssDepthProcessor(streamer.calib_lt)
    streamer.start()
    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        data_pv = data.data_pv
        data_lt = data.data_lt

        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, streamer.pv_intrinsics, streamer.pv_extrinsics)

        cv2.imshow('D',depth)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')