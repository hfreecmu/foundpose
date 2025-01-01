def convert_obj_to_mm(input_path, output_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()

    with open(output_path, 'w') as file:
        for line in lines:
            # Process only vertex lines
            if line.startswith('v '):
                parts = line.split()
                # Convert vertex coordinates from meters to millimeters
                x, y, z = map(float, parts[1:])
                x, y, z = x * 1000, y * 1000, z * 1000
                file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            else:
                # Write other lines unchanged
                file.write(line)

# Paths for the input and output OBJ files
input_file = '/home/hfreeman/Downloads/mesh_test/textured_mesh.obj'
output_file = '/home/hfreeman/Downloads/mesh_test/textured_mesh_mm.obj'

convert_obj_to_mm(input_file, output_file)