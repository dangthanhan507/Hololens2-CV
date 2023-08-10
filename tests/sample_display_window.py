import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_render import Hl2ssRender, RenderObject
from render_lib import RenderBBox, setMultiObjectPose, CoordinateFrame, ClassDisplayWindow
from pynput import keyboard
import time
from detector import BBox3D
from scipy.spatial.transform import Rotation as R
import numpy as np

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


def main(render, streamer):
    frame = CoordinateFrame([0,0,0],size=1e-1)
    frame_objs = frame.create_render()
    frame_ids = render.addPrimObjects(frame_objs)

    bbox3d = BBox3D(-0.1,-0.1,-0.1,0.1,0.1,0.1,'bbox')
    renderbbox = RenderBBox(bbox3d, thickness=0.01)
    bbox3d_objs = renderbbox.create_render()

    bbox3d_objs_ids = render.addPrimObjects(bbox3d_objs)

    center_sphere = render.addPrimObject(RenderObject('sphere',[0,0,0],[0,0,0,1],[1e-2]*3,[1,1,1,1]))

    window = ClassDisplayWindow(bbox3d)
    window_objs = window.create_render()
    window_ids = render.addPrimObjects(window_objs)

    print(window_objs)

    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        pv_pose = data.data_pv.pose.T

        x = pv_pose[0,-1]
        z = pv_pose[2,-1]

        angle = np.arctan2(z,x)
        rot_mat = R.from_rotvec(np.array([0,-angle - np.pi/2,0])).as_matrix()

        pose = np.eye(4)
        pose[:3,:3] = rot_mat

        window_objs = setMultiObjectPose(window_objs, window.objs_pose, pose)

        render.transformObjs(window_ids, window_objs)
        time.sleep(1)

if __name__ == '__main__':
    print('Starting up Server')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    streamer = Hl2ssStreamWrapper()
    render = Hl2ssRender()
    streamer.start()
    render.start()
    render.clear()
    try:
        main(render,streamer)
    except:
        type, value, traceback = sys.exc_info()
        print(traceback)
        print(value)

    listener.join()
    render.clear()
    render.stop()
    streamer.stop()

    print('Finishing up Server')