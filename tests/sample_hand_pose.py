import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_render import Hl2ssRender, RenderObject
from pynput import keyboard
import cv2

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


class HandRenderer:
    def __init__(self,renderer):
        self.renderer = renderer

        self.lefthand_ids = []
        self.righthand_ids = []

        self.lefthand_objs = []
        self.righthand_objs = []
    def visualize_lefthand(self, left_pos):
        N = left_pos.shape[0]
        if len(self.lefthand_ids) == 0:
            for i in range(N):
                pos = left_pos[i,:].tolist()
                # print(pos)
                rotation = [0,0,0,1]
                scale = [0.01,0.01,0.01]
                rgba = [1,1,1,1]
                pos[2] *= -1

                self.lefthand_objs.append(RenderObject('sphere',pos,rotation,scale,rgba))
            self.lefthand_ids = self.renderer.addPrimObjects(self.lefthand_objs)
        else:
            for i in range(N):
                pos = left_pos[i,:].tolist()
                pos[2] *= -1
                self.lefthand_objs[i].pos = pos
            self.renderer.transformObjs(self.lefthand_ids,self.lefthand_objs)
    def visualize_righthand(self, right_pos):
        N = right_pos.shape[0]
        if len(self.righthand_ids) == 0:
            for i in range(N):
                pos = right_pos[i,:].tolist()
                # print(pos)
                rotation = [0,0,0,1]
                scale = [0.01,0.01,0.01]
                rgba = [1,1,1,1]
                pos[2] *= -1

                self.righthand_objs.append(RenderObject('sphere',pos,rotation,scale,rgba))
            self.righthand_ids = self.renderer.addPrimObjects(self.righthand_objs)
        else:
            for i in range(N):
                pos = right_pos[i,:].tolist()
                pos[2] *= -1
                self.righthand_objs[i].pos = pos
            self.renderer.transformObjs(self.righthand_ids,self.righthand_objs)
if __name__ == '__main__':
    print('Starting up Server')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    streamer = Hl2ssStreamWrapper(opts={"spatial_input": True})
    render = Hl2ssRender()
    streamer.start()
    render.start()
    render.clear()

    hand_vis = HandRenderer(render)
    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        lt = data.data_lt.payload.depth
        pv = data.data_pv.payload.image
        if not (data.data_si.si is None):
            if not (data.data_si.hand_left is None):
                left_pos = data.data_si.hand_left.positions
                print(left_pos)
                print()

                hand_vis.visualize_lefthand(left_pos)

            if not (data.data_si.hand_right is None):
                right_pos = data.data_si.hand_right.positions
                # print(right_pos)
                hand_vis.visualize_righthand(right_pos)


        cv2.imshow('LT',lt / lt.max())
        cv2.imshow('PV',pv[:,:,::-1])
        cv2.waitKey(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')