import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy

def interpolate_poses(start_pose, end_pose, N):
    assert start_pose.shape == (3, 4)
    assert end_pose.shape == (3, 4)
    
    # Decompose the poses into rotation and translation parts
    start_rot = start_pose[:3, :3]
    start_trans = start_pose[:3, 3]
    end_rot = end_pose[:3, :3]
    end_trans = end_pose[:3, 3]
    
    # Convert rotation matrices to quaternions
    start_rot_q = torch.tensor(scipy.spatial.transform.Rotation.from_matrix(start_rot.numpy()).as_quat(), dtype=torch.float32)
    end_rot_q = torch.tensor(scipy.spatial.transform.Rotation.from_matrix(end_rot.numpy()).as_quat(), dtype=torch.float32)

    intermediate_poses = []
    
    for i in range(1, N + 1):
        t = i / (N + 1)  # Normalized interpolation factor

        # Lerp for translation
        inter_trans = start_trans * (1 - t) + end_trans * t

        # Slerp for rotation
        inter_rot_q = torch.lerp(start_rot_q, end_rot_q, t)
        inter_rot_q = inter_rot_q / inter_rot_q.norm()  # Ensure the quaternion stays normalized
        inter_rot = torch.tensor(scipy.spatial.transform.Rotation.from_quat(inter_rot_q.numpy()).as_matrix(), dtype=torch.float32)
        
        # Reconstruct the pose matrix
        inter_pose = torch.cat((inter_rot, inter_trans.unsqueeze(-1)), dim=-1)
        intermediate_poses.append(inter_pose)
    
    return intermediate_poses


# Define two 3x4 pose matrices
start_pose = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3]], dtype=torch.float32)
end_pose = torch.tensor([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, -1, 6]], dtype=torch.float32)

# Number of intermediate poses
N = 5

# Interpolate poses
intermediate_poses = interpolate_poses(start_pose, end_pose, N)

# Print the start pose, end pose, and interpolated poses
print("Start pose:")
print(start_pose.numpy())

for i, pose in enumerate(intermediate_poses, start=1):
    print(f"\nIntermediate pose {i}:")
    print(pose.numpy())

print("\nEnd pose:")
print(end_pose.numpy())