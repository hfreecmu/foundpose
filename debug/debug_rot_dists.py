import os
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def rotation_distance(rot1, rot2) -> float:
    rot1 = Rotation.from_matrix(rot1)
    rot2 = Rotation.from_matrix(rot2)

    # Compute the relative rotation
    relative_rotation = rot1.inv() * rot2
    
    # Convert to angle-axis to get the geodesic distance
    angle = relative_rotation.magnitude()
    return angle

def ensure_quaternion_continuity(quaternions):
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i], quaternions[i - 1]) < 0:
            quaternions[i] *= -1
    return quaternions

# def filter_outliers(data, max_iter=100):
#     outliers = np.array([False]*len(data))
#     for _ in range(max_iter):
#         q1, q3 = np.percentile(data[~outliers], [25, 75])
#         iqr = q3 - q1
#         thresh = q3 + 1.5 * iqr

#         new_outliers = (data > thresh) and (data != np.inf)
#         new_outliers[1:] = np.bitwise_and(new_outliers, ~new_outliers[0:-1])

#         if new_outliers.sum() == 0:
#             return outliers

#         outliers = np.bitwise_or(outliers, new_outliers)
#         data[outliers] = np.inf

#         breakpoint()


DATA_DIR = 'output/sugar/inference/lmo_v1/pruners'

filenames = []
for filename in os.listdir(DATA_DIR):
    if not filename.endswith('.txt'):
        continue

    filenames.append(filename)

filenames = sorted(filenames)

ts = []
quats = []
for filename in filenames:
    M = np.loadtxt(os.path.join(DATA_DIR, filename))
    R = M[0:3, 0:3]
    quat = Rotation.from_matrix(R).as_quat()

    ts.append(M[0:3, 3])
    quats.append(quat)

quats = ensure_quaternion_continuity(quats)

# smoothed_translations = savgol_filter(ts, 13, 2, axis=0)
# smoothed_quaternions = savgol_filter(quats, 13, 2, axis=0)
smoothed_translations = np.copy(ts)
smoothed_quaternions = np.copy(quats)

smoothed_quaternions = smoothed_quaternions / np.linalg.norm(smoothed_quaternions, axis=1)[:, None]

smoothed_poses = []
for st, sq in zip(smoothed_translations, smoothed_quaternions):
    sR = Rotation.from_quat(sq).as_matrix()

    smoothed_pose = np.eye(4)
    smoothed_pose[0:3, 0:3] = sR
    smoothed_pose[0:3, 3] = st

    smoothed_poses.append(smoothed_pose)

smoothed_poses = np.array(smoothed_poses)

dists = []
for ind in range(1, smoothed_poses.shape[0]):
    prev_R = smoothed_poses[ind - 1, 0:3, 0:3]
    curr_R = smoothed_poses[ind, 0:3, 0:3]

    dist = rotation_distance(prev_R, curr_R) * 180.0 / np.pi
    dists.append(dist)

dists = np.array(dists)

q1, q3 = np.percentile(dists, [25, 75])
iqr = q3 - q1
thresh = q3 + 1.5 * iqr

plt.plot(dists)
plt.axhline(y=thresh, color='orange', linestyle='--', linewidth=2)
plt.show()



# Rs = []
# for filename in filenames:
#     M = np.loadtxt(os.path.join(DATA_DIR, filename))
#     R = M[0:3, 0:3]

#     Rs.append(R)

# dists = []
# for ind in range(1, len(Rs)):
#     prev_R = Rs[ind - 1]
#     curr_R = Rs[ind]

#     dist = rotation_distance(prev_R, curr_R) * 180.0 / np.pi
#     dists.append(dist)

# dists = np.array(dists)

# # filter_outliers(dists)

# q1, q3 = np.percentile(dists, [25, 75])
# iqr = q3 - q1
# thresh = q3 + 1.5 * iqr

# plt.plot(dists)
# plt.axhline(y=thresh, color='orange', linestyle='--', linewidth=2)
# plt.show()
