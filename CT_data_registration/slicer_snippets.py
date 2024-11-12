rawMeshNode = slicer.util.getNode("run2_vslam_map_centered")
vertex_points = vtk.util.numpy_support.vtk_to_numpy(rawMeshNode.GetPolyData().GetPoints().GetData())
points = vertex_points

vtk_points = vtk.vtkPoints()
vtk_points.SetData(vtk.util.numpy_support.numpy_to_vtk(points))

# Create the vtkPolyData object.
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)

# Create the vtkSphereSource object.
sphere = vtk.vtkSphereSource()
sphere.SetRadius(0.1)

# Create the vtkGlyph3D object.
glyph = vtk.vtkGlyph3D()
glyph.SetInputData(polydata)
glyph.SetSourceConnection(sphere.GetOutputPort())
pointCloudModelNode = slicer.modules.models.logic().AddModel(glyph.GetOutputPort())