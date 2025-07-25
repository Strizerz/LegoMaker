import trimesh

path = "src\\Car.obj"
#load the model
mesh = trimesh.load(path)

#change the model from a scene to the first object which is usually the model 
mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

#quality or size of the voxels, lower value is higher quality
quality = 0.2

#turn it into a grid of voxels without a mesh
voxelGrid = mesh.voxelized(pitch=quality)

#positions of each voxels
voxelPos = voxelGrid.points

#define sizes for cylinders for studs
studRadius = quality / 3
studHeight = quality / 5

#initialize list of studs
studs = []

#for each voxel add a cylinder on top
for pos in voxelPos:
    stud = trimesh.creation.cylinder(radius=studRadius, height=studHeight, sections=16)
    stud.apply_translation([pos[0], pos[1], pos[2] + quality / 2 + studHeight / 2])
    studs.append(stud)

#combine the cubes with cylinders to create studs
allMeshes = [mesh] + studs
combined = trimesh.util.concatenate(allMeshes)

#show the studs
combined.show()



