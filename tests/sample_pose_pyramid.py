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


class Pose:
    def __init__(self, rotm, tvec):
        self.rot_mat = rotm
        self.t_vec = tvec

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
    def __init__(self, sphere_rad, bar_length):
        self.bar_length = bar_length
        self.sphere_rad = sphere_rad
        self.objs = None

        
        self.bar_pose    = Pose(np.eye(3), np.array([[0,0,bar_length]]).T )
        self.sphere_pose = Pose(np.eye(3), self.bar_pose.t_vec*2 + np.array([[0,0,sphere_rad/2]]).T )

        self.bar = None
        self.sphere = None

    def create_render(self):
        #elongated cube with a sphere on it default points towards +x
        pos = self.bar_pose.t_vec.flatten().tolist()
        rotation = R.from_matrix(self.bar_pose.rot_mat).as_quat().tolist()
        scale = [1e-2,1e-2,self.bar_length]
        rgba = [1,1,1,1]
        self.bar = RenderObject('cube', pos, rotation, scale, rgba)


        pos = self.sphere_pose.t_vec.flatten().tolist()
        rotation = R.from_matrix(self.sphere_pose.rot_mat).as_quat().tolist()
        scale = [self.sphere_rad]*3
        rgba = [1,1,1,1]
        self.sphere = RenderObject('sphere', pos, rotation, scale, rgba)

        self.objs = [self.bar,self.sphere]
        return self.objs
    def set_pose(self, pose):
        rot_mat = pose[:3,:3]
        t_vec = pose[:3,-1].reshape((3,1))

        Rot_world = rot_mat
        t_world = t_vec

        Rot_bar = self.bar_pose.rot_mat
        t_bar   = self.bar_pose.t_vec

        pos = ((Rot_world @ t_bar) + t_world).flatten().tolist()
        rotation = R.from_matrix(Rot_world @ Rot_bar).as_quat()
        self.bar.pos = pos
        self.bar.rot = rotation

        Rot_sphere = self.sphere_pose.rot_mat
        t_sphere   = self.sphere_pose.t_vec

        pos = ((Rot_world @ t_sphere) + t_world).flatten().tolist()
        rotation = R.from_matrix(Rot_world @ Rot_sphere).as_quat()
        self.sphere.pos = pos
        self.sphere.rot = rotation

        self.objs = [self.bar, self.sphere]

        


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


    pendulum = PosePendulum(1e-1,4*1e-1)
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
        pv_pose = data.data_pv.pose.T
        # flip_axis = np.eye(4)
        # flip_axis[2,2] = -1

        pv_pose[2,-1] *= -1
        pv_pose[0,:3] *= -1
        pv_pose[1,:3] *= -1


        pendulum.set_pose(pv_pose)


        render.transformObjs(pend_ids, pendulum.objs)
        # time.sleep(0.01)

    render.clear()
    render.stop()
    streamer.stop()
    listener.join()

    print('Finishing up Server')