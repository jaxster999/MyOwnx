import numpy as np

def read_obj(fileName):
    """
    Read wavefront OBJ models with or without textures.
    Supports triangles and quads (converted into triangles).
    Handles v, v/vt, v//vn, and v/vt/vn formats.
    """
    vertices, triangles, texture_uv, texture_map = [], [], [], []

    with open(fileName, "r") as f:
        for line in f.readlines():
            splitted = line.strip().split()

            if len(splitted) == 0:
                continue

            if splitted[0] == "v":
                vertices.append(splitted[1:4] + [1,1,1])

            elif splitted[0] == "vt":
                texture_uv.append(splitted[1:3])

            elif splitted[0] == "f":
                vert_indices = []
                tex_indices  = []

                for face_part in splitted[1:]:
                    parts = face_part.split("/")

                    # ---- Vertex index (always present)
                    vert_indices.append(int(parts[0]) - 1)

                    # ---- Texture index (may be missing)
                    if len(parts) > 1 and parts[1] != "":
                        tex_indices.append(int(parts[1]) - 1)
                    else:
                        tex_indices.append(-1)   # mark missing UV

                # Convert quad → 2 triangles
                if len(vert_indices) == 3:
                    triangles.append(vert_indices)
                    texture_map.append(tex_indices)

                elif len(vert_indices) == 4:
                    # triangle 1
                    triangles.append([vert_indices[0], vert_indices[1], vert_indices[2]])
                    texture_map.append([tex_indices[0], tex_indices[1], tex_indices[2]])
                    # triangle 2
                    triangles.append([vert_indices[0], vert_indices[2], vert_indices[3]])
                    texture_map.append([tex_indices[0], tex_indices[2], tex_indices[3]])

    vertices = np.asarray(vertices).astype(float)
    triangles = np.asarray(triangles).astype(int)

    if len(texture_uv) > 0 and np.any(np.array(texture_map) != -1):
        textured = True
        texture_uv = np.asarray(texture_uv).astype(float)
        texture_uv[:,1] = 1 - texture_uv[:,1]   # flip V coordinate
        texture_map = np.asarray(texture_map).astype(int)
    else:
        texture_uv = np.asarray(texture_uv)
        texture_map = np.asarray(texture_map)
        textured = False

    return vertices, triangles, texture_uv, texture_map, textured
