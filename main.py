import trimesh
import numpy as np
from trimesh.transformations import rotation_matrix

def greedy_pack_voxels(voxel_grid, inventory):
    matrix       = voxel_grid.matrix.copy()
    placements   = []
    brick_types  = sorted(inventory, key=lambda b: b[0]*b[1]*b[2], reverse=True)
    remaining    = inventory.copy()

    # iterate brick types from largest volume → smallest
    for dims in brick_types:
        dx, dy, dz = dims
        quota      = remaining[dims]
        if quota <= 0:
            continue
        nx, ny, nz = matrix.shape
        for i in range(nx-dx+1):
            for j in range(ny-dy+1):
                for k in range(nz-dz+1):
                    if quota <= 0:
                        break
                    if np.all(matrix[i:i+dx, j:j+dy, k:k+dz]):
                        placements.append({'dims': dims, 'origin': (i, j, k)})
                        matrix[i:i+dx, j:j+dy, k:k+dz] = False
                        quota     -= 1
                        remaining[dims] -= 1
                if quota <= 0:
                    break
            if quota <= 0:
                break
    return placements, remaining



# Brick‑type to RGBA colour map
airbrush = {
    (1, 1, 1): [255,   0,   0, 255],  # red
    (2, 4, 1): [  0,   0, 255, 255],  # blue
    (4, 1, 1): [  0, 255,   0, 255]   # green
}

# Global orientation transforms
g_orientation = {
    'front': rotation_matrix(0,           [1, 0, 0]),        # identity
    'back' : rotation_matrix(np.pi,       [0, 1, 0]),        # 180° Y
    'left' : rotation_matrix(np.pi/2,     [0, 1, 0]),        # +90° Y
    'right': rotation_matrix(-np.pi/2,    [0, 1, 0]),        # -90° Y
    'up'   : rotation_matrix(np.pi/2,     [1, 0, 0]),        # +90° X
    'down' : rotation_matrix(-np.pi/2,    [1, 0, 0])         # -90° X
}

# Cylinder default axis is +Z; rotate +90° about X so stud axis → +Y
rot_to_pos_y = rotation_matrix(np.pi/2, [1, 0, 0])


def visualize_placements(placements, grid, orientation='front'):
    """Build a Trimesh scene with coloured bricks & studs, then show it."""
    scene = trimesh.Scene()

    # pitch vector and min corner of voxel grid (world coords)
    px, py, pz        = grid.pitch
    xmin, ymin, zmin  = grid.bounds[0]
    global_R          = g_orientation.get(orientation, g_orientation['front'])

    for p in placements:
        dx, dy, dz = p['dims']
        i, j, k    = p['origin']

        # 1) Brick body 
        extents = np.array([dx*px, dz*pz, dy*py])
        centre  = np.array([
            xmin + (i + dx/2) * px,
            ymin + (k + dz/2) * pz,
            zmin + (j + dy/2) * py
        ])

        brick = trimesh.creation.box(extents=extents)
        brick.apply_translation(centre)
        brick.apply_transform(global_R)  # apply chosen orientation
        clr = airbrush.get((dx, dy, dz), [200, 200, 200, 255])
        brick.visual.vertex_colors = np.tile(clr, (brick.vertices.shape[0], 1))
        scene.add_geometry(brick)

        # 2) Studs 
        stud_rad = min(px, py) * 0.3
        stud_h   = py * 0.2
        for u in range(dx):
            for v in range(dy):
                stud = trimesh.creation.cylinder(radius=stud_rad, height=stud_h, sections=24)
                stud.apply_transform(rot_to_pos_y)  # axis → +Y before global
                # local offset relative to brick centre (before global_R)
                loff = np.array([
                    (u - (dx-1)/2) * px,
                    extents[1]/2 + stud_h/2,
                    (v - (dy-1)/2) * py
                ])
                stud.apply_translation(centre + loff)
                stud.apply_transform(global_R)      # same global orientation
                stud.visual.vertex_colors = np.tile(clr, (stud.vertices.shape[0], 1))
                scene.add_geometry(stud)

    scene.show()

def main():
    mesh_path = './src/Car.obj'      
    mesh      = trimesh.load(mesh_path, force='mesh')

    # choose voxel pitch (resolution) 
    pitch = mesh.bounding_box.extents.max() * 0.025   # 5 % of longest side
    grid  = mesh.voxelized(pitch=pitch)              # set fill=False for shell

    # inventory limits 
    inventory = {
        (2, 4, 1): 500,
        (4, 1, 1): 200,
        (1, 1, 1): 1000
    }

    # pack & show 
    placements, remainder = greedy_pack_voxels(grid, inventory)
    print(remainder)
    # orientation options: 'front', 'back', 'left', 'right', 'up', 'down'
    visualize_placements(placements, grid, orientation='front')


if __name__ == '__main__':
    main()
