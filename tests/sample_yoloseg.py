import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_stream import Hl2ssStreamWrapper, Hl2ssData
from detector import BBox, YoloDetector, YoloSegment, BBox3D
from hl2ss_render import Hl2ssRender, RenderObject
from hl2ss_utils import Hl2ssDepthProcessor
from render_lib import RenderBBox, setMultiObjectPose, CoordinateFrame
from pynput import keyboard
import cv2
import cv_utils
import numpy as np
import time

enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable


def main(render, streamer):
    depth_processor = Hl2ssDepthProcessor(streamer.calib_lt)

    rotation = [0, 0, 0, 1]
    scale = np.array([0.1,0.1,0.1])*0.8
    rgba = [1,1,0,1]
    detector = YoloSegment("yolov8n-seg.pt")

    while enable:
        streamer.waitReady()
        data = streamer.getData()
        if data is None:
            print('Skipped')
            continue
        render.clear()
        data_pv = data.data_pv
        data_lt = data.data_lt

        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        pts3d_image = cv_utils.rgbd_getpoints_imshape(depth, data.color_intrinsics[:3,:3].T)
        masks, bboxes = detector.eval(rgb)

        # frame = CoordinateFrame([0,0,0],size=1e-1)
        # frame_objs = frame.create_render()
        # frame_ids = render.addPrimObjects(frame_objs)

        depth_mask = (np.zeros(pts3d_image.shape[:2]) != 0)
        for n in range(len(masks)):
            mask = masks[n]
            mask = cv2.resize(mask,pts3d_image.shape[:2][::-1],interpolation=cv2.INTER_AREA)
            pts3d_mask = pts3d_image

            pts3d = pts3d_mask.reshape(3,-1)
            pts3d = pts3d[:,mask.flatten() > 0]
            pts3d = pts3d[:,pts3d[2,:] > 0]
            pts3d = np.vstack((pts3d, np.ones((1,pts3d.shape[1]))))
            pts_3d = (data_pv.pose.T @ np.linalg.inv(data.color_extrinsics.T) @ pts3d)[:3,:]
            
            #get info for 3d bboxs
            bbox3d = cv_utils.bbox_3d_from_pcd(pts_3d,name='bbox')
            renderbbox = RenderBBox(bbox3d,thickness=0.01)
            bbox3d_objs = renderbbox.create_render()
            bbox3d_objs_ids = render.addPrimObjects(bbox3d_objs)

            #render cube in middle of "bbox"
            pts_3d = np.mean(pts_3d,axis=1).reshape(3,1)

            pos = pts_3d.flatten().tolist()
            pos[2] *= -1
            render.addPrimObject(RenderObject("sphere", pos, rotation, scale.tolist(), rgba))

            bbox = bboxes[n]
            rgb = bbox.drawBox(rgb)

            depth_mask[mask > 0] = True
        

        depth[~depth_mask] = 0




        cv2.imshow('D',depth)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(1)
    return 0



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