import trimesh

path = "src\Car.obj"

mesh = trimesh.load(path)

mesh.show()