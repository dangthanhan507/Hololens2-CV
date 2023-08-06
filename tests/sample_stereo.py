import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_stereo import Hl2ssStereo
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

    stereo = Hl2ssStereo()
    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        lf = data.data_lf.payload
        rf = data.data_rf.payload

        lf, rf = stereo.stereo_pipeline(lf,rf)
        image = stereo.make_stereo_display(lf,rf)

        cv2.imshow('Display',image)
        cv2.waitKey(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')