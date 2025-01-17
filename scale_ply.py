import trimesh

def scale_ply(input_path, output_path):
    # Load the PLY mesh
    mesh = trimesh.load(input_path)

    # Scale the mesh by 1000
    mesh.apply_scale(1000)

    # Save the scaled mesh
    mesh.export(output_path, vertex_normal=True)

# Paths for the input and output OBJ files
input_file = '/home/hfreeman/harry_ws/repos/gs2mesh/output/sugar/custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p/pruners_colmap/pruners_colmap_custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask1_occ0_scale1_0_voxel0.001_512_trunc4_20_cleaned_mesh.ply'
output_file = '/home/hfreeman/harry_ws/repos/gs2mesh/output/sugar/custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p/pruners_colmap/pruners_colmap_custom_nw_iterations30000_DLNR_Middlebury_baseline7_0p_mask1_occ0_scale1_0_voxel0.001_512_trunc4_20_cleaned_mesh_mm.ply'

scale_ply(input_file, output_file)