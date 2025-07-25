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

#turn the voxel grid back to a mesh to visualize
voxelMesh = voxelGrid.as_boxes()

#show the model
voxelMesh.show()
