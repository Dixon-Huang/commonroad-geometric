from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import numpy as np
from commonroad_geometric.dataset.commonroad_data import CommonRoadData
from commonroad_geometric.dataset.postprocessing.base_data_postprocessor import BaseDataPostprocessor
from commonroad_geometric.simulation.base_simulation import BaseSimulation
from commonroad_geometric.simulation.ego_simulation.ego_vehicle import EgoVehicle
from commonroad_geometric.common.torch_utils.geometry import contains_any_rotated_rectangles


def compute_vectorized_occupancy(data: CommonRoadData, view_range: float = 70.0, grid_size: int = 64, is_polar: bool = False):
    device = data.device
    num_nodes = data.v.num_nodes
    cx = data.v.pos[:, 0] # (num_nodes,)
    cy = data.v.pos[:, 1] # (num_nodes,)
    width = data.v.width # (num_nodes, 1)
    height = data.v.length # (num_nodes, 1)
    angle = data.v.orientation # (num_nodes, 1)

    if is_polar:
        # Prepare the polar grid
        radial_bins, angular_bins = grid_size, grid_size
        max_radius = view_range / 2
        radii = torch.linspace(0, max_radius, radial_bins, device=device)
        angles = torch.linspace(-np.pi, np.pi, angular_bins, device=device)
        rr, aa = torch.meshgrid(radii, angles, indexing='ij')  # rr is radius, aa is angle

        # Convert polar grid to Cartesian coordinates centered at (0, 0)
        xx = rr * torch.cos(aa)
        yy = rr * torch.sin(aa)

    else:
        # Step 1: Create normalized grid coordinates
        x = torch.linspace(-1, 1, grid_size, device=device)
        y = torch.linspace(-1, 1, grid_size, device=device)
        xx, yy = torch.meshgrid(x, y)  # These will have shape (grid_size, grid_size)

        # Step 2: Scale grid (without shift)
        xx = xx * (view_range / 2)
        yy = yy * (view_range / 2)


    # Prepare for batch operations
    cos_orientations = torch.cos(angle)
    sin_orientations = torch.sin(angle)

    # Step 3: Apply rotation about the origin (vectorized)
    rotated_xx = cos_orientations[:, None, None] * xx - sin_orientations[:, None, None] * yy
    rotated_yy = sin_orientations[:, None, None] * xx + cos_orientations[:, None, None] * yy

    # Step 4: Shift grid by view_center (vectorized)
    rotated_xx += data.v.pos[:, 0][:, None, None, None]
    rotated_yy += data.v.pos[:, 1][:, None, None, None]
 
    rotation_matrix_to_global = torch.cat((cos_orientations, sin_orientations, -sin_orientations, cos_orientations), dim=1).view(angle.shape[0], 2, 2)

    # Calculate the velocities in the global frame
    global_velocities = torch.einsum('ijk,ik->ij', rotation_matrix_to_global, data.v.velocity)

    # Prepare to calculate relative velocities in local frames of each vehicle A
    # Matrix of global velocities for all A to B comparisons
    velocity_matrix = global_velocities.repeat(angle.shape[0], 1).view(angle.shape[0], angle.shape[0], 2)

    # Relative velocities in global frame
    relative_velocity_matrix = velocity_matrix - global_velocities.unsqueeze(1)

    # Rotation matrix to transform from global frame back to each vehicle A's local frame
    rotation_matrix_to_local = torch.cat((cos_orientations, -sin_orientations, sin_orientations, cos_orientations), dim=1).view(angle.shape[0], 2, 2)

    # Apply rotation to convert relative velocities to the local frame of vehicle A
    local_relative_velocity_matrix = torch.einsum('ijk,ilk->ijl', rotation_matrix_to_local, relative_velocity_matrix)
    local_relative_velocity_matrix_x = local_relative_velocity_matrix[:, 0, :]
    local_relative_velocity_matrix_y = local_relative_velocity_matrix[:, 1, :]

    local_grid_occupancy_matrix = contains_any_rotated_rectangles(
        x=rotated_xx.flatten(start_dim=1),  # Flatten all but the first dimension
        y=rotated_yy.flatten(start_dim=1),  # Flatten all but the first dimension
        cx=cx, 
        cy=cy, 
        width=width + 0.5*view_range/grid_size, 
        height=height + 0.5*view_range/grid_size, 
        angle=angle,
        reduce=False
    ).view(num_nodes, grid_size, grid_size, num_nodes)

    # Removing ego vehicles
    indices = torch.arange(local_grid_occupancy_matrix.shape[0], device=device) 
    local_grid_occupancy_matrix[indices, :, :, indices] = 0

    if 'is_clone' in data.v:
        vmask = ~(data.v.is_clone == 1).squeeze(-1)
        local_grid_occupancy_matrix = local_grid_occupancy_matrix[:, :, :, vmask]
    else:
        vmask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    local_occupancy_flow_matrix_x = local_grid_occupancy_matrix * local_relative_velocity_matrix_x[:, None, None, vmask]
    local_occupancy_flow_matrix_y = local_grid_occupancy_matrix * local_relative_velocity_matrix_y[:, None, None, vmask]
    
    local_grid_occupancy = local_grid_occupancy_matrix.max(-1)[0].float()
    local_occupancy_flow_x = local_occupancy_flow_matrix_x.sum(-1)
    local_occupancy_flow_y = local_occupancy_flow_matrix_y.sum(-1)

    return local_grid_occupancy, local_occupancy_flow_x, local_occupancy_flow_y


class VectorizedOccupancyPostProcessor(BaseDataPostprocessor):
    def __init__(
        self
    ) -> None:
        super().__init__()

    def __call__(
        self,
        samples: List[CommonRoadData],
        simulation: Optional[BaseSimulation] = None,
        ego_vehicle: Optional[EgoVehicle] = None
    ) -> List[CommonRoadData]:
        for data in samples:      
            occupancy, occupancy_flow_x, occupancy_flow_y = compute_vectorized_occupancy(data=data)
            polar_occupancy, polar_occupancy_flow_x, polar_occupancy_flow_y = compute_vectorized_occupancy(data=data, is_polar=True)

            data.v.occupancy = occupancy
            data.v.occupancy_flow = torch.stack([occupancy_flow_x, occupancy_flow_y], dim=-1)
            data.v.polar_occupancy = polar_occupancy
            data.v.polar_occupancy_flow = torch.stack([polar_occupancy_flow_x, polar_occupancy_flow_y], dim=-1)
            
        return samples

