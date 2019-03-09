import os,time,cv2, sys, math
import tensorflow as tf

import argparse
import numpy as np

## DB
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.filters import median
from skimage.morphology import disk

@adapt_rgb(each_channel)
def median_each(image, disksize=10):
    return median(image, disk(disksize))
    
    
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels, unary_from_softmax

# =========================================================
def getCRF(image, Lc, theta, n_iter, N, compat_spat=12, compat_col=40, scale=5, prob=0.5):

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

      d = dcrf.DenseCRF2D(H, W, N+1)
      U = unary_from_labels(Lc.astype('int'), N+1, gt_prob= prob)

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


compat_col = 300
theta = 300
scale = 1
n_iter = 30
compat_spat =11
prob = 0.5 
    
## DB 

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='', required=True, help='The image you want to predict on. ')
parser.add_argument('--checkpoint_path', type=str, default='', required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default='custom', help='The model you are using')
parser.add_argument('--dataset', type=str, default="Geomorph_B", required=False, help='The dataset you are using')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)

with tf.device('/cpu:0'):

    # Initializing network
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess=tf.Session(config=config)
    
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)

    net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
    net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

    network, _ = model_builder.build_model(args.model, net_input=net_input,
                                            num_classes=num_classes,
                                            crop_width=args.crop_width,
                                            crop_height=args.crop_height,
                                            is_training=False)

    sess.run(tf.global_variables_initializer())

    print('Loading model checkpoint weights')
    saver=tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, args.checkpoint_path)


    print("Testing image " + args.image)

    loaded_image = utils.load_image(args.image)
    resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
    input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_time = time.time()-st

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)

    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
   
    ##DB 
    mask1 = ((out_vis_image[:,:,0]==0).astype('int') + (out_vis_image[:,:,1]==0).astype('int')  + (out_vis_image[:,:,2]==255).astype('int')  == 3).astype('int')

    #sand
    mask2 = ((out_vis_image[:,:,0]==255).astype('int') + (out_vis_image[:,:,1]==255).astype('int')  + (out_vis_image[:,:,2]==0).astype('int')  == 3).astype('int')

    ##veg
    mask3 = ((out_vis_image[:,:,0]==0).astype('int') + (out_vis_image[:,:,1]==255).astype('int')  + (out_vis_image[:,:,2]==0).astype('int')  == 3).astype('int')

    #br
    mask4 = ((out_vis_image[:,:,0]==128).astype('int') + (out_vis_image[:,:,1]==0).astype('int')  + (out_vis_image[:,:,2]==0).astype('int')  == 3).astype('int')

    #df
    mask5 = ((out_vis_image[:,:,0]==255).astype('int') + (out_vis_image[:,:,1]==0).astype('int')  + (out_vis_image[:,:,2]==255).astype('int')  == 3).astype('int')

    #at
    mask6 = ((out_vis_image[:,:,0]==128).astype('int') + (out_vis_image[:,:,1]==0).astype('int')  + (out_vis_image[:,:,2]==255).astype('int')  == 3).astype('int')

    #null
    mask7 = ((out_vis_image[:,:,0]==0).astype('int') + (out_vis_image[:,:,1]==0).astype('int')  + (out_vis_image[:,:,2]==0).astype('int')  == 3).astype('int')

    #gb
    mask8 = ((out_vis_image[:,:,0]==128).astype('int') + (out_vis_image[:,:,1]==128).astype('int')  + (out_vis_image[:,:,2]==128).astype('int')  == 3).astype('int')

    #other
    mask8 = ((out_vis_image[:,:,0]==255).astype('int') + (out_vis_image[:,:,1]==128).astype('int')  + (out_vis_image[:,:,2]==128).astype('int')  == 3).astype('int')
    
    nx, ny, nz = np.shape(out_vis_image)
    lab = np.zeros((nx, ny), dtype=np.int)
    lab[mask1==1]=1
    lab[mask2==1]=2
    lab[mask3==1]=3
    lab[mask4==1]=4
    lab[mask5==1]=5
    lab[mask6==1]=6
    lab[mask7==1]=7
    lab[mask8==1]=8   
    
    
    idx = np.random.randint(nx, size=int(nx/2))
    for k in idx:
       lab[:,k] = np.zeros(ny)
       lab[k,:] = np.zeros(ny)
   
    N = 9 ##number categories
    
    resr = getCRF(loaded_image, lab, theta, n_iter, N, compat_spat, compat_col, scale, prob)+1
 
    resr = median_each(resr, disksize=3)  ##DB
    resr = getCRF(loaded_image, resr, theta, n_iter, N, compat_spat, compat_col, scale, prob)+1
            
    out_vis_image = np.zeros((nx, ny, 3), dtype=np.int)
    #water
    mask = (resr==1).astype('int')
    out_vis_image[:,:,0][mask==1]=0
    out_vis_image[:,:,1][mask==1]=0
    out_vis_image[:,:,2][mask==1]=255

    #sand
    mask = (resr==2).astype('int')
    out_vis_image[:,:,0][mask==1]=255
    out_vis_image[:,:,1][mask==1]=255
    out_vis_image[:,:,2][mask==1]=0

    #veg
    mask = (resr==3).astype('int')
    out_vis_image[:,:,0][mask==1]=0
    out_vis_image[:,:,1][mask==1]=255
    out_vis_image[:,:,2][mask==1]=0

    #br
    mask = (resr==4).astype('int')
    out_vis_image[:,:,0][mask==1]=128
    out_vis_image[:,:,1][mask==1]=0
    out_vis_image[:,:,2][mask==1]=0

    #df
    mask = (resr==5).astype('int')
    out_vis_image[:,:,0][mask==1]=255
    out_vis_image[:,:,1][mask==1]=0
    out_vis_image[:,:,2][mask==1]=255

    #at
    mask = (resr==6).astype('int')
    out_vis_image[:,:,0][mask==1]=128
    out_vis_image[:,:,1][mask==1]=0
    out_vis_image[:,:,2][mask==1]=255

    #null
    mask = (resr==7).astype('int')
    out_vis_image[:,:,0][mask==1]=0
    out_vis_image[:,:,1][mask==1]=0
    out_vis_image[:,:,2][mask==1]=0

    #gb
    mask = (resr==8).astype('int')
    out_vis_image[:,:,0][mask==1]=128
    out_vis_image[:,:,1][mask==1]=128
    out_vis_image[:,:,2][mask==1]=128     

    #other
    mask = (resr==8).astype('int')
    out_vis_image[:,:,0][mask==1]=255
    out_vis_image[:,:,1][mask==1]=128
    out_vis_image[:,:,2][mask==1]=128  
 

            
    ##DB    
        
    file_name = utils.filepath_to_name(args.image)
    cv2.imwrite("%s_pred.png"%(file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

    print("")
    print("Finished!")
    print("Wrote image " + "%s_pred.png"%(file_name))
