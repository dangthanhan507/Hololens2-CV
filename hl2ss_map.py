from hl2ss_stream import HOST_IP, ROOT_PATH
import os
import sys
sys.path.append(os.path.join(ROOT_PATH,'hl2ss','viewer'))

import hl2ss
import hl2ss_3dcv
import hl2ss_sa
import numpy as np
import open3d as o3d

# mesh setting
TPCM = 1_000 # Triangles per cubic meter
VPF = hl2ss.SM_VertexPositionFormat.R32G32B32A32Float
TIF = hl2ss.SM_TriangleIndexFormat.R32Uint
VNF = hl2ss.SM_VertexNormalFormat.R32G32B32A32Float
USE_NORMALS = True # compute normals?
THREADS = 2 # HoloLens threads to compute meshes

class Hl2ssMapping:
    def __init__(self, download=True):
        self.sm_manager = None
        if download:
            self.sm_manager = hl2ss_sa.sm_manager(HOST_IP, TPCM, THREADS)
        else:
            pass



    def observe_map(self):
        sm_origin = [0, 0, 0] # Origin of sampling volume
        sm_radius = 100 # Radius of sampling volume 
        sm_volumes = hl2ss.sm_bounding_volume()
        sm_volumes.add_sphere(sm_origin, sm_radius)

        self.sm_manager.open()
        self.sm_manager.set_volumes(sm_volumes)
        self.sm_manager.get_observed_surfaces()
        self.sm_manager.close()

        self.meshes = self.sm_manager.get_meshes()

    def get_o3d_mesh(self):
        # meshes = self.sm_manager.get_meshes()
        meshes = self.meshes

        o3d_mesh = o3d.geometry.TriangleMesh()
        for mesh in meshes:
            open3d_mesh = hl2ss_3dcv.sm_mesh_to_open3d_triangle_mesh(mesh)
            open3d_mesh.vertex_colors = open3d_mesh.vertex_normals

            o3d_mesh += open3d_mesh
        return [o3d_mesh]
    
    def save_map(self, o3d_mesh, path, name):
        o3d.io.write_triangle_mesh(filename=os.path.join(path,f'{name}.obj'),mesh=o3d_mesh,write_ascii=True,write_vertex_normals=True,write_vertex_colors=True,write_triangle_uvs=True)

    def load_map(self, path):
        return o3d.io.read_triangle_mesh(path)
