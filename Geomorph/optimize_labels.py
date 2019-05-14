

from __future__ import division

import sys, getopt, os
import time, socket 

if sys.version[0]=='3':
   from tkinter import Tk, Toplevel 
   from tkinter.filedialog import askopenfilename
   import tkinter
   import tkinter as tk
   from tkinter.messagebox import *   
   from tkinter.filedialog import *
else:
   from Tkinter import Tk, TopLevel
   from tkFileDialog import askopenfilename
   import Tkinter as tkinter
   import Tkinter as tk
   from Tkinter.messagebox import *   
   from Tkinter.filedialog import *   

#import cv2
import numpy as np
from scipy.misc import imsave, imread, imresize 
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from numpy.lib.stride_tricks import as_strided as ast
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels, unary_from_softmax
from matplotlib import cm

#import os.path as path
#from skimage.filters.rank import median
#from skimage.morphology import disk

from imageio import imwrite
from scipy.stats import mode as md
from skimage.color import label2rgb
from numpy.lib.stride_tricks import as_strided as ast

# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')


# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape
	
# =========================================================
def getCRF(image, Lc, theta, n_iter, label_lines, compat_spat=12, compat_col=40, scale=5, prob=0.5):

#        n_iters: number of iterations of MAP inference.
#        sxy_gaussian: standard deviations for the location component
#            of the colour-independent term.
#        compat_gaussian: label compatibilities for the colour-independent
#            term (can be a number, a 1D array, or a 2D array).
#        kernel_gaussian: kernel precision matrix for the colour-independent
#            term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#        normalisation_gaussian: normalisation for the colour-independent term
#            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#        sxy_bilateral: standard deviations for the location component of the colour-dependent term.
#        compat_bilateral: label compatibilities for the colour-dependent
#            term (can be a number, a 1D array, or a 2D array).
#        srgb_bilateral: standard deviations for the colour component
#            of the colour-dependent term.
#        kernel_bilateral: kernel precision matrix for the colour-dependent term
#            (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#        normalisation_bilateral: normalisation for the colour-dependent term
#            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

      H = image.shape[0]
      W = image.shape[1]

      d = dcrf.DenseCRF2D(H, W, len(label_lines)+1)
      U = unary_from_labels(Lc.astype('int'), len(label_lines)+1, gt_prob= prob)

      d.setUnaryEnergy(U)

      del U

      # This potential penalizes small pieces of segmentation that are
      # spatially isolated -- enforces more spatially consistent segmentations
      # This adds the color-independent term, features are the locations only.
      # sxy = The scaling factors per dimension.
      d.addPairwiseGaussian(sxy=(theta,theta), compat=compat_spat, kernel=dcrf.DIAG_KERNEL, #compat=6
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

      # sdims = The scaling factors per dimension.
      # schan = The scaling factors per channel in the image.
      # This creates the color-dependent features and then add them to the CRF
      feats = create_pairwise_bilateral(sdims=(theta, theta), schan=(scale, scale, scale), #11,11,11
                                  img=image, chdim=2)

      del image

      d.addPairwiseEnergy(feats, compat=compat_col, #20
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)
      del feats

      Q = d.inference(n_iter)

	  ## uncomment if you want an images of the posterior probabilities per label
      #preds = np.array(Q, dtype=np.float32).reshape(
      #  (len(label_lines)+1, nx, ny)).transpose(1, 2, 0)
      #preds = np.expand_dims(preds, 0)
      #preds = np.squeeze(preds)

      return np.argmax(Q, axis=0).reshape((H, W)) #, preds


#medfiltsize = 5
win = 5	
	
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
image_path = askopenfilename(filetypes=[("pick an image file","*.JPG *.jpg *.jpeg *.JPEG *.png *.PNG *.tif *.tiff *.TIF *.TIFF")], multiple=True)  
   
labels_path = askopenfilename(filetypes=[("pick a labels file","*.txt")], multiple=False)  
colors_path = askopenfilename(filetypes=[("pick a label colors file","*.txt")], multiple=False)  
   	  
mat_files = [k.split('.')[0]+'_mres.mat' for k in image_path]

labels = loadmat(mat_files[0])['labels']
labels = [x.strip() for x in labels] 

dat = []
for k in mat_files:
   dat.append(loadmat(k)['sparse'])

ims = []
for k in image_path:
   ims.append(imread(k))   

with open(colors_path) as f: #'labels.txt') as f:
   cols = f.readlines()
cmap1 = [x.strip() for x in cols] 
 
classes = dict(zip(labels, cmap1))

cmap = colors.ListedColormap(cmap1)   
   
## cycle through 60 settings

for data_counter in range(len(ims)):

   print('working on '+str(image_path[data_counter]))
   R = []
  
   for theta in [40, 80, 160, 240, 320]:
      print(theta)
      for compat_spat in [1,5,10]:
         for compat_col in [30,60,90,120]:
            R.append(getCRF(ims[data_counter], dat[data_counter].astype('int'), theta, 20, labels, compat_spat, compat_col, 1, 0.51))

   r, cnt = md(R)
   r = np.squeeze(r)+1
   r[np.squeeze(cnt)<=30] = 0 ##if count is less than half number of settings, call it zero
   
   ##relabel blue channel pixels < 5 the code associated with 'null'
   r[ims[data_counter][:,:,2]<5] = int(np.where(np.asarray(labels)=='null')[0])+1

   r = 1+getCRF(ims[data_counter], r, 60, 20, labels, 1, 60, 1, 0.51)

   #rm = median(r, disk(medfiltsize))-1
   nx, ny = np.shape(r)
   Z,ind = sliding_window(r, (win, win), (win, win))
   gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))
   Zx,_ = sliding_window(gridx, (win, win), (win, win))
   Zy,_ = sliding_window(gridy, (win, win), (win, win))
   
   rm = np.zeros((nx,ny))
   c = np.zeros((nx,ny))   
   for ck in range(len(Zx)):
      tmp, cnt = md(Z[ck])
      rm[Zx[ck],Zy[ck]] = np.squeeze(tmp)
      c[Zx[ck],Zy[ck]] = np.squeeze(cnt)
   rm[c<=(win-1)] = 0	  
   rm = getCRF(ims[data_counter], rm, 10, 20, labels, 1, 10, 1, 0.9)
   
   name, ext = os.path.splitext(image_path[data_counter])
   name = name.split(os.sep)[-1]     
   #=============================================   
   #=============================================
   print('Generating plot ....')
   fig = plt.figure()
   fig.subplots_adjust(wspace=0.4)
   ax1 = fig.add_subplot(121)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(ims[data_counter])
   #plt.title('a) Input', loc='left', fontsize=6)

   ax1 = fig.add_subplot(122)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)   

   _ = ax1.imshow(ims[data_counter])
   #plt.title('c) CRF prediction', loc='left', fontsize=6)
   im2 = ax1.imshow(rm, cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(0.5+np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)
   plt.savefig(name+'_mresc.png', dpi=600)#, bbox_inches='tight')
   del fig; plt.close()

   savemat(image_path[data_counter].split('.')[0]+'_mresc.mat', {'sparse': dat[data_counter].astype('int'), 'class': rm.astype('int'), 'preds': np.nan, 'labels': labels}, do_compression = True) 

   rgb = []
   for k in cmap1:
      rgb.append(tuple(int(k.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))

   out = label2rgb(rm.astype('int')+1, image=None, colors=[rgb[k] for k in np.unique(rm)], bg_color=(0, 0, 0), image_alpha=1, kind='overlay')

   imwrite(name+'_mres_label.png', out.astype('uint8'))



	  