import os
import numpy as np

data_dir = '/home/hfreeman/harry_ws/repos/feature-3dgs/data/pruners/ob_in_cam'
ts = []
for filename in os.listdir(data_dir):
    M = np.loadtxt(os.path.join(data_dir, filename))
    t = M[0:3, 3]

    ts.append(t)

ts = np.array(ts)
zs = ts[:, -1] 

print('mean', zs.mean())
print('median', np.median(zs))
print('min, max', zs.min(), zs.max())
print('range_avg', (zs.min() + zs.max()) / 2)