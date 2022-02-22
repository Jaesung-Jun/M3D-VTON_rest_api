from plyfile import PlyData, PlyElement
import sys
with open('../../user_requests/test33334/results/custom_model.ply', 'rb') as f:
    plydata = PlyData().read(f)
plydata.write('test.ply')