import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


# pcd = o3d.io.read_point_cloud("cuppcd/1_outlier_removed.ply")


def do_surface_construction():
    pcd = o3d.io.read_point_cloud("merged_multiview_fullscene.ply")

    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(100)

    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)


    ply_filename = "mesh_colorized.ply"
    o3d.io.write_triangle_mesh(ply_filename, mesh, write_ascii=True)
    print(f"Colorized Mesh was written to {ply_filename}")

    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=False, mesh_show_back_face=True)

    densities = np.asarray(densities)

    #mapping the densities to colors using plasma colormap
    density_colors = plt.get_cmap("plasma")(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]

    #for visualization, this is a new mesh with the same vertices and triangles as the original mesh
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    #assigning the colors to the new mesh based on the densities
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)


    o3d.visualization.draw_geometries([density_mesh], mesh_show_wireframe=False, mesh_show_back_face=True)

    #removing the vertices with the lowest 1% of densities
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=False, mesh_show_back_face=True)
    o3d.io.write_triangle_mesh("mesh_colorized_filtered.ply", mesh, write_ascii=True)

    # references used: open3d documentation