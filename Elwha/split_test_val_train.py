

import os, shutil, glob
import numpy as np

direcI = os.getcwd()+'/R_G_B_tiles'
direcL = os.getcwd()+'/label_tiles'

allfilesI = sorted(glob.glob(direcI+os.sep+'*.tif'))
allfilesL = sorted(glob.glob(direcL+os.sep+'*.tif'))

prop_train = 0.5
prop_test = 0.75

print(str(len(allfilesI))+' image files found')  
print(str(len(allfilesL))+' label files found')  

##train
idx = np.random.choice(np.arange(len(allfilesI)), int(prop_train*len(allfilesI)), replace=False)  

tomove = [allfilesI[k] for k in idx] 
print('moving '+str(len(tomove))+' files')  
for f in tomove:
   shutil.move(f, os.getcwd()+'/train')	

tomove = [allfilesL[k] for k in idx] 

for f in tomove:
   shutil.move(f, os.getcwd()+'/train_labels')	


##test
allfilesI = sorted(glob.glob(direcI+os.sep+'*.tif'))
allfilesL = sorted(glob.glob(direcL+os.sep+'*.tif'))

print(str(len(allfilesI))+' image files found')  
print(str(len(allfilesL))+' label files found')  

idx = np.random.choice(np.arange(len(allfilesI)), int(prop_test*len(allfilesI)), replace=False)  

tomove = [allfilesI[k] for k in idx] 
print('moving '+str(len(tomove))+' files')  
for f in tomove:
   shutil.move(f, os.getcwd()+'/test')	

tomove = [allfilesL[k] for k in idx]  

for f in tomove:
   shutil.move(f, os.getcwd()+'/test_labels')	


##val
tomoveI = sorted(glob.glob(direcI+os.sep+'*.tif'))
tomoveL = sorted(glob.glob(direcL+os.sep+'*.tif'))

print('moving '+str(len(tomoveI))+' files')  
for f in tomoveI:
   shutil.move(f, os.getcwd()+'/val')	

for f in tomoveL:
   shutil.move(f, os.getcwd()+'/val_labels')	


