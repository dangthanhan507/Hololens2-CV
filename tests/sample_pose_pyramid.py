import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_render import Hl2ssRender, RenderObject
from pynput import keyboard
import time

from scipy.spatial.transform import Rotation as R
import numpy as np

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


class CoordinateFrame:
    def __init__(self, offset, size):
        self.offset = offset
        self.size = size

    def create_render(self):
        #draw coordinate frame
        size = self.size

        pos = self.offset.copy()
        pos[0] += size/2
        rotation = [0,0,0,1]
        scale = [size,1e-2,1e-2]
        rgba = [1,0,0,1]

        xaxis = RenderObject('cube', pos, rotation, scale, rgba)

        pos = self.offset.copy()
        pos[1] += size/2
        scale = [1e-2,size,1e-2]
        rgba = [0,1,0,1]
        yaxis = RenderObject('cube', pos, rotation, scale, rgba)

        pos = self.offset.copy()
        pos[2] += size/2
        scale = [1e-2,1e-2,size]
        rgba = [0,0,1,1]
        zaxis = RenderObject('cube', pos, rotation, scale, rgba)

        return [xaxis,yaxis,zaxis]

class PosePendulum:
    def __init__(self, offset):
        self.offset = offset
        self.objs = None

    def create_render(self):
        #elongated cube with a sphere on it default points towards +x
        size = 4*1e-1
        pos = self.offset.copy()
        pos[0] += size/2
        rotation = [0,0,0,1]
        scale = [size,1e-2,1e-2]
        rgba = [1,1,1,1]

        bar = RenderObject('cube', pos, rotation, scale, rgba)

        pos = self.offset.copy()
        pos[0] += size + 1e-1/2
        scale = [1e-1,1e-1,1e-1]
        sphere = RenderObject('sphere', pos, rotation, scale, rgba)

        self.objs = [bar,sphere]
        return self.objs
    def set_rotmatrix(self, rot):
        Rot = R.from_matrix(rot)

        rotation = Rot.as_quat()
        rotm = Rot.as_matrix()
        orig_pos = np.array(self.offset.copy())

        bar_opos = [size/2,0,0]
        bar_pos = np.array((rotm @ np.array(bar_opos).reshape((3,1))).flatten() + orig_pos)

        for obj in self.objs:
            obj.rot = rotation
            obj.pos = bar_pos
        


class PosePyramid:
    def __init__(self, pose):
        '''
            pose: (4x4)
        '''
        self.rot_mat = pose[:3,:3]
        self.t       = pos[:3,-1]

    def create_render(self):
        pass

if __name__ == '__main__':
    print('Starting up Server')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    render = Hl2ssRender()
    render.start()
    render.clear()


    #draw coordinate frame
    size = 5*1e-1
    offset = [0,0,2]

    cf = CoordinateFrame(offset,size)
    cf_objs = cf.create_render()

    render.addPrimObjects(cf_objs)

    offset = [0,0,1]
    pendulum = PosePendulum(offset)
    pend_objs = pendulum.create_render()

    pend_ids = render.addPrimObjects(pend_objs)

    streamer = Hl2ssStreamWrapper()
    streamer.start()

    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        pv_pose = data.data_pv.pose
        print(pv_pose.T)

        pendulum.set_rotmatrix(pv_pose.T[:3,:3])
        render.transformObjs(pend_ids, pendulum.objs)
        time.sleep(1)

    render.clear()
    render.stop()
    streamer.stop()
    listener.join()

    print('Finishing up Server')