import sys
ROOT_PATH = "/home/andang/workspace/CV_Lab/Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)
from hl2ss_map import Hl2ssMapping
import open3d as o3d

if __name__ == '__main__':
    mapping = Hl2ssMapping()

    mapping.observe_map()
    meshes = mapping.get_o3d_mesh()
    o3d.visualization.draw_geometries(meshes, mesh_show_back_face=True)