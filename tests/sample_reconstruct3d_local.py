import copy
import pickle
import sys

ROOT_PATH = "../Hololens2-CV-Server/"
sys.path.append(ROOT_PATH)

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

if __name__ == '__main__':
    with open("all_3d_pts.pkl", "rb") as f:
        all_3d_pts = pickle.load(f)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_3d_pts.T)
    # pcd.estimate_normals()
    # assert pcd.has_normals()

    # remove outlier pc
    # alpha: voxel_size=0.02, ball pivoting mesh: voxel_size=0.01
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02) # 0.01
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    voxel_down_pcd = cl.select_by_index(ind)
    voxel_down_pcd.estimate_normals()
    voxel_down_pcd.orient_normals_consistent_tangent_plane(100)
    assert voxel_down_pcd.has_normals()

    # Alpha Mesh
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(voxel_down_pcd, alpha=0.03)
    # mesh.compute_vertex_normals()
    # mesh.filter_smooth_laplacian(500)

    # Ball Pivoting Mesh
    # radii = [0.04, 0.04, 0.04, 0.04]
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #     voxel_down_pcd, o3d.utility.DoubleVector(radii))

    # Poisson Mesh
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            voxel_down_pcd, depth=15)
    print(mesh)
    vertices_to_remove = densities < np.quantile(densities, 0.75)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([1, 0.706, 0])

    # remove outliers
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 250
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw_geometries([mesh_0, voxel_down_pcd])
