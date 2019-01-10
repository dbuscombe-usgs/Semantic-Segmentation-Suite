from imageio import imread
import numpy as np
from glob import glob
import os

T0 = []
T1 = []
T2 = []
labeltiles = glob(os.getcwd()+'label_tiles/*.tif')
for file in labeltiles:
   im = imread(file)
   T0.append(sorted(np.unique(im[:,:,0])))   
   T1.append(sorted(np.unique(im[:,:,1])))   
   T2.append(sorted(np.unique(im[:,:,2])))   

codes = np.column_stack((np.unique(np.hstack(T0)), np.unique(np.hstack(T1)), np.unique(np.hstack(T2))))      
      
print(codes)

