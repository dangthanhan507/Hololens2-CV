


import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_render import Hl2ssRender, RenderObject
from render_lib import RenderBBox, setMultiObjectPose, CoordinateFrame
from pynput import keyboard
import time
from detector import BBox3D

import numpy as np

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


if __name__ == '__main__':
    print('Starting up Server')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    render = Hl2ssRender()
    render.start()
    render.clear()

    frame = CoordinateFrame([0,0,0],size=1e-1)
    frame_objs = frame.create_render()
    frame_ids = render.addPrimObjects(frame_objs)

    bbox3d = BBox3D(-0.1,-0.1,-0.1,0.1,0.1,0.1,'bbox')
    renderbbox = RenderBBox(bbox3d, thickness=0.01)
    bbox3d_objs = renderbbox.create_render()

    bbox3d_objs_ids = render.addPrimObjects(bbox3d_objs)

    center_sphere = render.addPrimObject(RenderObject('sphere',[0,0,0],[0,0,0,1],[1e-2]*3,[1,1,1,1]))

    while enable:
        pass

    render.clear()
    render.stop()
    listener.join()

    print('Finishing up Server')