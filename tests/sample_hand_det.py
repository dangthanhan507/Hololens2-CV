import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from hl2ss_render import Hl2ssRender, RenderObject
from pynput import keyboard
import cv2
from detector import YoloSegment, preprocess_bbox_IOU
from hl2ss_utils import Hl2ssDepthProcessor
from hl_sensorstack import HololensSensorStack, KinematicChain
from multi_object_tracker import MultiObjectTracker
import cv_utils
import numpy as np

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

    detector = YoloSegment("yolov8n-seg.pt")
    kin_chain = KinematicChain(streamer.pv_intrinsics, streamer.pv_extrinsics)
    sensor_stack = HololensSensorStack()
    depth_processor = Hl2ssDepthProcessor(sensor_stack.calib_lt)
    tracker = MultiObjectTracker()
    
    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None or data.data_pv is None or data.data_lt is None:
            print('Skipped')
            continue
        kin_chain.update_pv_calibration(data.color_intrinsics.T, data.color_extrinsics.T)
        sensor_stack.update_pv_calibration(data.color_intrinsics.T, data.color_extrinsics.T)
        data_pv = data.data_pv
        data_lt = data.data_lt


        #get spatial input data
        left_pos = None
        right_pose = None
        if not (data.data_si.si is None):
            if not (data.data_si.hand_left is None):
                left_pos = data.data_si.hand_left.positions

            if not (data.data_si.hand_right is None):
                right_pos = data.data_si.hand_right.positions


        #get rgbd
        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        pts3d_image = cv_utils.rgbd_getpoints_imshape(depth, data.color_intrinsics[:3,:3].T)
        

        #perform detection
        bboxes = []
        masks, boxes = detector.eval(rgb,filter_cls=['cup'])
        for n in range(len(boxes)):
            mask = masks[n]
            mask = cv2.resize(mask,pts3d_image.shape[:2][::-1],interpolation=cv2.INTER_AREA)
            pts3d_mask = pts3d_image

            pts3d = pts3d_mask.reshape(3,-1)
            pts3d = pts3d[:,mask.flatten() > 0]
            pts3d = pts3d[:,pts3d[2,:] > 0]
            pts3d = np.vstack((pts3d, np.ones((1,pts3d.shape[1]))))
            pts_3d = (data_pv.pose.T @ np.linalg.inv(data.color_extrinsics.T) @ pts3d)[:3,:]

            if left_pos is not None:
                #NOTE: 0.1 is a good distance away from object

                norm = np.linalg.norm(left_pos.T.mean(axis=1) - pts_3d.mean(axis=1))
                print('norm:', norm)

                if norm < 0.1:
                    boxes[n].name = boxes[n].name + ' interacting'

            rgb = boxes[n].drawBox(rgb)


        cv2.imshow("RGB", rgb)
        cv2.waitKey(1)

    streamer.stop()
    listener.join()

    print('Finishing up Server')