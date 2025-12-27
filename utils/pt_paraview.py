import torch
import vtk
from vtk.util import numpy_support
import os
import numpy as np

def convert_torch_to_paraview_2d(tensor_2d: torch.Tensor, filename: str = "tensor_2d.vti"):
    if tensor_2d.ndim != 2:
        raise ValueError(f"Input tensor must be 2D, but got {tensor_2d.ndim}D")

    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    numpy_array = tensor_2d.detach().cpu().numpy()
    numpy_array = np.flipud(numpy_array)
    numpy_array = numpy_array.astype(np.float32)

    nx, ny = numpy_array.shape[1], numpy_array.shape[0]

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, 1)
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)

    flat_array = numpy_array.flatten(order='C')
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=flat_array, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data_array.SetName("values")
    image_data.GetPointData().SetScalars(vtk_data_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()
    print(f"Saved 2D tensor to {filename} using vtk")

def convert_torch_to_paraview_3d(tensor_3d: torch.Tensor, filename: str = "tensor_3d.vti"):
    if tensor_3d.ndim != 3:
        raise ValueError(f"Input tensor must be 3D, but got {tensor_3d.ndim}D")

    # Check and create directory if it doesn't exist
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    numpy_array = tensor_3d.detach().cpu().numpy() # Shape: (D, H, W)

    data_for_vtk = numpy_array.transpose(2, 1, 0) # Transpose to (W, H, D)

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(data_for_vtk.shape[0], data_for_vtk.shape[1], data_for_vtk.shape[2]) # W, H, D
    image_data.SetSpacing(1.0, 1.0, 1.0) # Optional
    image_data.SetOrigin(0.0, 0.0, 0.0)  # Optional

    vtk_data_array = numpy_support.numpy_to_vtk(num_array=data_for_vtk.flatten(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
    vtk_data_array.SetName("values")
    
    image_data.GetPointData().SetScalars(vtk_data_array)

    # Write to .vti file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()
    print(f"Saved 3D tensor to {filename} using vtk")
