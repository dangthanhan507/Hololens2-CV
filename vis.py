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
import numpy as np

enable = True
def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def Rotx(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta), -np.sin(theta) ],
                    [0,         np.sin(theta), np.cos(theta)  ]
                    ])
    return R_x

def Roty(theta):
    R_y = np.array([[np.cos(theta),    0,      np.sin(theta)  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta),   0,      np.cos(theta)  ]
                    ])
    return R_y

def Rotz(theta):
    R_z = np.array([[np.cos(theta),    -np.sin(theta),    0],
                    [np.sin(theta),    np.cos(theta),     0],
                    [0,                     0,                      1]
                    ])
    return R_z

def pad_4x4(matrix):
    # turn 3x3 into 4x4 matrix
    trans = np.zeros((3, 1), dtype=matrix.dtype)
    matrix = np.concatenate((matrix, trans), axis=-1) # 3x4
    bottom = np.array([0, 0, 0, 1], dtype=matrix.dtype).reshape(1, 4)
    matrix = np.concatenate((matrix, bottom), axis=0)
    return matrix

if __name__ == '__main__':
    print('Starting up Player')
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


    player = Hl2ssOfflineStreamer('./offline_folder/offline_script2',{})
    player.open()
    
    view_width_px = 640
    view_height_px = 360

    intrinsic = np.array([320, 0.0, 320, 0.0, 180, 180, 0., 0., 1.0], dtype=np.float64).reshape(3, 3)

    mapper = Hl2ssMapping(download=False)
    render_option_path = 'renderoption.json'
    o3d_mesh = mapper.load_map('./offline_folder/offline_script2/map.obj')
    o3d_mesh.compute_vertex_normals()

    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(o3d_mesh.triangles)[:,::-1])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(o3d_mesh)
    


    while enable:
        data = player.getData()
        if data is None:
            print('Skipped')
            continue
        else:
            print('Play')

        data_pv = data.data_pv
        pose = data_pv.pose.T
        
        flip = Rotx(np.pi)

        pose[:3,:3] = flip @ pose[:3,:3]
        
        extrinsic = np.linalg.inv(pose)
        cam_obj = o3d.geometry.LineSet.create_camera_visualization(view_width_px, view_height_px, intrinsic, extrinsic, scale=0.25)

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 10  # note that this is sc

        vis.add_geometry(cam_obj)
        vis.get_render_option().load_from_json(render_option_path)
        ctr = vis.get_view_control()
        camera_params = ctr.convert_to_pinhole_camera_parameters()

        T = np.array([
            [ 1., 0., 0., 1.7],
            [ 0., 1., 0., 10.],
            [ 0., 0., 1, 1.64],
            [ 0., 0., 0., 1.]], dtype=np.float32).reshape(4, 4)
        camera_params.extrinsic = np.linalg.inv(pad_4x4(Roty(np.pi/10))@T@pad_4x4(Rotx(np.pi/2))@pad_4x4(Roty(np.pi/50))@pad_4x4(Rotx(np.pi/50)))
        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(False)
        time.sleep(0.2)
        vis.remove_geometry(cam_obj, False)

    player.close()
    listener.join()
    print('Finishing up Player')