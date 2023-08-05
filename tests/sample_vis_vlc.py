import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
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

    streamer = Hl2ssStreamWrapper(opts={"vlc_front": True})
    streamer.start()
    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        lt = data.data_lt.payload.depth
        pv = data.data_pv.payload.image
        lf = data.data_lf.payload
        rf = data.data_rf.payload


        cv2.imshow('LF',lf)
        cv2.imshow('RF',rf)
        cv2.waitKey(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')