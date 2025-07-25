import trimesh 
import numpy as np

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

def createStuds(voxelGrid, quality, studRadius, studHeight):
    #positions of each voxels
    voxelPos = voxelGrid.points

    #initialize list of studs
    studs = []

    #for each voxel add a cylinder on top
    for pos in voxelPos:
        stud = trimesh.creation.cylinder(radius=studRadius, height=studHeight, sections=16)

        #rotate the cylinders
        stud.apply_transform(trimesh.transformations.rotation_matrix(
            angle=np.pi / 2,
            direction=[1, 0, 0],
            point=[0, 0, 0]
        ))
        stud.apply_translation([pos[0], pos[1] + quality / 2 + studHeight / 2, pos[2]])
        
        studs.append(stud)

    return studs

def main():
    path = "src\\Car.obj"

    mesh = loadModel(path)

    #quality or size of the voxels, lower value is higher quality
    quality = 0.2

    voxelGrid, voxelMesh = voxelizeMesh(mesh, quality)

    #define sizes for cylinders for studs
    studRadius = quality / 3
    studHeight = quality / 4

    studs = createStuds(voxelGrid, quality, studRadius, studHeight)

    #combine the cubes with cylinders to create studs
    allMeshes = [voxelMesh] + studs
    combined = trimesh.util.concatenate(allMeshes)

    #show the studs
    combined.show()


if __name__ == "__main__":
    main()
