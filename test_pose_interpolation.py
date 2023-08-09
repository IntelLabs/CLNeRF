import torch
import scipy.spatial.transform

def interpolate_poses(start_pose, end_pose, N):
    assert start_pose.shape == (3, 4)
    assert end_pose.shape == (3, 4)

    start_rot = start_pose[:3, :3]
    start_trans = start_pose[:3, 3]
    end_rot = end_pose[:3, :3]
    end_trans = end_pose[:3, 3]

    start_rot_q = torch.tensor(scipy.spatial.transform.Rotation.from_matrix(start_rot.numpy()).as_quat(), dtype=torch.float32)
    end_rot_q = torch.tensor(scipy.spatial.transform.Rotation.from_matrix(end_rot.numpy()).as_quat(), dtype=torch.float32)

    # Ensure the quaternions have the shortest arc for interpolation
    if torch.dot(start_rot_q, end_rot_q) < 0:
        end_rot_q = -end_rot_q

    intermediate_poses = []

    for i in range(1, N + 1):
        t = i / (N + 1)  # Normalized interpolation factor

        # Lerp for translation
        inter_trans = start_trans * (1 - t) + end_trans * t

        # Slerp for rotation
        inter_rot_q = torch.lerp(start_rot_q, end_rot_q, t)
        inter_rot_q /= inter_rot_q.norm()
        inter_rot = torch.tensor(scipy.spatial.transform.Rotation.from_quat(inter_rot_q.numpy()).as_matrix(), dtype=torch.float32)

        inter_pose = torch.cat((inter_rot, inter_trans.unsqueeze(-1)), dim=-1)
        intermediate_poses.append(inter_pose)
    
    return intermediate_poses

# # Usage example:
# start_pose = torch.eye(3, 4)
# end_pose = torch.eye(3, 4)
# end_pose[0, 3] = 1.0

# interpolated_poses = interpolate_poses(start_pose, end_pose, 5)

# for pose in interpolated_poses:
#     print(pose)
import torch
import numpy as np
import scipy.spatial.transform

# Define the start and end poses
start_pose = torch.eye(3, 4)
end_rot = scipy.spatial.transform.Rotation.from_euler('z', 30, degrees=True).as_matrix()
end_pose = torch.eye(3, 4)
end_pose[:3, :3] = torch.tensor(end_rot, dtype=torch.float32)
end_pose[0, 3], end_pose[1, 3], end_pose[2, 3] = 1.0, 1.0, 1.0

# Interpolation function
# ... (as defined previously)

# Generate interpolated poses
interpolated_poses = interpolate_poses(start_pose, end_pose, 5)

# Verification steps
def angle_from_quaternion_dot(dot_val):
    """Compute the angle of rotation based on the dot product of two quaternions."""
    return 2 * np.arccos(dot_val)

# 1. Verify Translation
differences = []
for i in range(1, len(interpolated_poses)):
    trans_diff = torch.norm(interpolated_poses[i][:3, 3] - interpolated_poses[i-1][:3, 3]).item()
    differences.append(trans_diff)

print("Translation differences between consecutive poses:", differences)
print("Are translations consistently interpolated?", len(set(differences)) == 1)

# 2. Verify Rotation
angles = []
start_rot_q = scipy.spatial.transform.Rotation.from_matrix(start_pose[:3, :3].numpy()).as_quat()
for pose in interpolated_poses:
    interp_rot_q = scipy.spatial.transform.Rotation.from_matrix(pose[:3, :3].numpy()).as_quat()
    dot_product = np.dot(start_rot_q, interp_rot_q)
    angle = angle_from_quaternion_dot(dot_product)
    angles.append(np.degrees(angle))  # Convert angle from radians to degrees

print("\nAngles (in degrees) from start pose to each interpolated pose:", angles)
print("Are all angles <= 180 degrees?", all(angle <= 180 for angle in angles))
