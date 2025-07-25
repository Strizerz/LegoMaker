import trimesh 
import numpy as np
from collections import Counter

def loadModel(path):
    #load the model
    mesh = trimesh.load(path)
    #change the model from a scene to the first object which is usually the model 
    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh

def voxelizeMesh(mesh, quality):
    #turn it into a grid of voxels without a mesh
    voxelGrid = mesh.voxelized(pitch=quality)
    voxelMesh = voxelGrid.as_boxes()
    return voxelGrid, voxelMesh

def createStuds(voxelGrid, quality, studRadius, studHeight, colors):

    #positions of each voxels
    voxelPos = voxelGrid.points

    #initialize list of studs
    studs = []

    #for each voxel add a cylinder on top
    global placements
    for origin, size in placements:
        sx, sy, sz = size
        sy, sz = sz, sy
        for ix in range(sx):
            for iy in range(sz):
                cx = origin[0] + (ix + 0.5) * quality
                cz = origin[2] + (iy + 0.5) * quality
                top_y = origin[1] + sy * quality
                stud = trimesh.creation.cylinder(radius=studRadius, height=studHeight, sections=8)
                stud.apply_transform(trimesh.transformations.rotation_matrix(
                    angle=np.pi / 2,
                    direction=[1, 0, 0],
                    point=[0, 0, 0]
                ))
                stud.apply_translation([cx, top_y + studHeight / 2, cz])
                color = colors.get(size, [200, 200, 200, 255])
                stud.visual.face_colors = color
                studs.append(stud)

    return studs

def Builder(voxelGrid, pitch, inventory, colors):

    #convert voxelPos to grid
    points = voxelGrid.points
    min_corner = np.floor(points.min(axis=0) / pitch).astype(int)
    max_corner = np.ceil(points.max(axis=0) / pitch).astype(int)
    grid_shape = tuple(max_corner - min_corner + 1)

    grid = np.zeros(grid_shape, dtype=bool)
    filled = np.zeros(grid_shape, dtype=bool)

    #mark where the voxels are
    for p in points:
        i = tuple(((p / pitch).round().astype(int)) - min_corner)
        grid[i] = True

    placements = []
    bricks = []

    #sort brick types largest to smallest || greedy algorithm approach
    sortedBricks = sorted(inventory.keys(), key=lambda x: (x[0]*x[1]*x[2]), reverse=True)

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                if not grid[x, y, z] or filled[x, y, z]:
                    continue

                for size in sortedBricks:
                    origSize = size
                    sx, sy, sz = size
                    sy, sz = sz, sy
                    if inventory[origSize] <= 0:
                        continue

                    # Check bounds
                    if x + sx > grid.shape[0] or y + sy > grid.shape[1] or z + sz > grid.shape[2]:
                        continue

                    # Check if brick fits and all voxels are present
                    region = grid[x:x+sx, y:y+sy, z:z+sz]
                    region_filled = filled[x:x+sx, y:y+sy, z:z+sz]
                    if np.all(region) and not np.any(region_filled):
                        # Place the brick
                        filled[x:x+sx, y:y+sy, z:z+sz] = True
                        inventory[origSize] -= 1

                        # Convert back to world coordinates
                        origin = (np.array([x, y, z]) + min_corner) * pitch
                        placements.append((origin, origSize))

                        # Create a box mesh and color it
                        box = trimesh.creation.box(extents=np.array([sx, sy, sz]) * pitch)
                        box.apply_translation(origin + np.array([sx, sy, sz]) * pitch / 2)
                        color = colors.get(origSize, [200, 200, 200, 255])
                        box.visual.face_colors = color
                        bricks.append(box)

                        break

    combined = trimesh.util.concatenate(bricks)
    return placements, combined


def createOutlineVisible(placements, pitch, color=[0,0,0,255]):
    segs = []
    # build occupancy set in grid coords
    occupied = set()
    for origin, size in placements:
        ix0, iy0, iz0 = (origin / pitch).astype(int)
        sx, sy, sz = size
        sy, sz = sz, sy
        for dx in range(sx):
            for dy in range(sy):
                for dz in range(sz):
                    occupied.add((ix0+dx, iy0+dy, iz0+dz))

    # face definitions with corners scaled to brick size
    face_defs = [
        ((0, 1, 0), lambda sx, sy, sz: [(0, sy, 0), (sx, sy, 0), (sx, sy, sz), (0, sy, sz)]),
        ((0,-1, 0), lambda sx, sy, sz: [(0, 0, 0), (0, 0, sz), (sx, 0, sz), (sx, 0, 0)]),
        ((1, 0, 0), lambda sx, sy, sz: [(sx,0,0), (sx,0,sz), (sx,sy,sz), (sx,sy,0)]),
        ((-1,0, 0), lambda sx, sy, sz: [(0,0,0), (0,sy,0), (0,sy,sz), (0,0,sz)]),
        ((0, 0, 1), lambda sx, sy, sz: [(0,0,sz), (0,sy,sz), (sx,sy,sz), (sx,0,sz)]),
        ((0, 0,-1), lambda sx, sy, sz: [(0,0,0), (sx,0,0), (sx,sy,0), (0,sy,0)])
    ]

    for origin, size in placements:
        ix0, iy0, iz0 = (origin / pitch).astype(int)
        sx, sy, sz = size
        sy, sz = sz, sy
        for offset, corner_fn in face_defs:
            ox, oy, oz = offset
            # check neighbor occupancy
            skip = False
            if ox != 0:
                nx = ix0 + (sx if ox > 0 else -1)
                for dy in range(sy):
                    for dz in range(sz):
                        if (nx, iy0+dy, iz0+dz) in occupied:
                            skip = True; break
                    if skip: break
            elif oy != 0:
                ny = iy0 + (sy if oy > 0 else -1)
                for dx in range(sx):
                    for dz in range(sz):
                        if (ix0+dx, ny, iz0+dz) in occupied:
                            skip = True; break
                    if skip: break
            else:
                nz = iz0 + (sz if oz > 0 else -1)
                for dx in range(sx):
                    for dy in range(sy):
                        if (ix0+dx, iy0+dy, nz) in occupied:
                            skip = True; break
                    if skip: break
            if skip:
                continue

            # draw the 4 edges of this face
            corners = corner_fn(sx * pitch, sy * pitch, sz * pitch)
            pts = [(origin[0] + cx, origin[1] + cy, origin[2] + cz) for cx, cy, cz in corners]
            for i in range(4):
                segs.append((pts[i], pts[(i+1) % 4]))

    if not segs:
        return None

    seg_array = np.array([[a, b] for a, b in segs])
    path = trimesh.load_path(seg_array)
    path.colors = np.tile(color, (len(path.entities), 1)).astype(np.uint8)
    return path

def createStudOutlines(studs, segments=12, color=[0,0,0,255]):
    outlines = []
    for stud in studs:
        # get center and axis (stud was rotated π/2 around X, so axis is +Y)
        center = stud.centroid
        r = stud.radius if hasattr(stud, 'radius') else None
        # fallback: compute radius from bounding box XZ extents
        if r is None:
            ext = stud.bounds
            r = (ext[1][0] - ext[0][0]) / 2
        # top and bottom y
        y0 = center[1] - stud.extents[1] / 2
        y1 = center[1] + stud.extents[1] / 2
        pts_top = []
        pts_bot = []
        for i in range(segments):
            theta = 2 * np.pi * i / segments
            x = center[0] + r * np.cos(theta)
            z = center[2] + r * np.sin(theta)
            pts_top.append((x, y1, z))
            pts_bot.append((x, y0, z))
        # connect into segments
        segs = []
        for i in range(segments):
            nxt = (i + 1) % segments
            segs.append((pts_top[i], pts_top[nxt]))
            segs.append((pts_bot[i], pts_bot[nxt]))
        path = trimesh.load_path(np.array(segs))
        path.colors = np.tile(color, (len(path.entities), 1)).astype(np.uint8)
        outlines.append(path)
    return outlines


def main():
    global placements
    path = "src\\Car.obj"

    mesh = loadModel(path)

    #quality or size of the voxels, lower value is higher quality
    quality = 0.2

    voxelGrid, voxelMesh = voxelizeMesh(mesh, quality)

    #define available bricks
    inventory = {
        (1, 4, 1): 200,
        (1, 2, 1): 400,
        (2, 4, 1): 300,
        (2, 1, 2): 200,
        (1, 1, 1): 9999
        
    }

    #define color for each brick size
    colors = {
        (1, 4, 1): [0, 0, 255, 255],
        (1, 2, 1): [0, 255, 255, 255],
        (2, 4, 1): [0, 255, 0, 255],
        (2, 1, 2): [255, 255, 0, 255],
        (1, 1, 1): [255, 0, 0, 255],
    }

    placements, rebuilt = Builder(voxelGrid, quality, inventory, colors)

    #define sizes for cylinders for studs
    studRadius = quality / 3
    studHeight = quality / 4

    studs = createStuds(voxelGrid, quality, studRadius, studHeight, colors)

    #combine the rebuilt brick model with the studs
    combined = trimesh.util.concatenate([rebuilt] + studs)

    #outline of bricks
    outline = createOutlineVisible(placements, quality, [0,0,0,255])

    #outline of studs
    stud_outlines = createStudOutlines(studs)

    scene = trimesh.Scene()
    scene.add_geometry(combined)
    if outline is not None:
        scene.add_geometry(outline)
    for o in stud_outlines:
        scene.add_geometry(o)
    scene.show()


if __name__ == "__main__":
    main()
