import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

from hl2ss_read import Hl2ssOfflineStreamer
from pynput import keyboard
from hl2ss_utils import Hl2ssDepthProcessor
from hl_sensorstack import HololensSensorStack
from hl2ss_map import Hl2ssMapping
import cv2
import time
import open3d as o3d

enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

if __name__ == '__main__':
    print('Starting up Player')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    mapper = Hl2ssMapping(download=False)
    o3d_mesh = mapper.load_map('./offline_folder/offline_script2/map.obj')
    o3d.visualization.draw_geometries([o3d_mesh], mesh_show_back_face=True)

    player = Hl2ssOfflineStreamer('./offline_folder/offline_script2',{})
    player.open()

    sensor_stack = HololensSensorStack()
    depth_processor = Hl2ssDepthProcessor(sensor_stack.calib_lt)
    
    while enable:
        data = player.getData()
        if data is None:
            print('Skipped')
            continue

        data_pv = data.data_pv
        data_lt = data.data_lt
        rgb, depth = depth_processor.create_rgbd(data_lt, data_pv, data.color_intrinsics, data.color_extrinsics)
        print(depth.max())
        cv2.imshow('D',depth)
        cv2.imshow('PV',rgb)
        cv2.waitKey(1)
        time.sleep(0.02)

    player.close()
    listener.join()
    print('Finishing up Player')