from imageio import imread
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
    
root = '/home/filfy/github_clones/Semantic-Segmentation-Suite'    
    
tiles_use = ['/B_tile_1536_14848_pred.png','/C_tile_2048_8192_pred.png', '/C_tile_3072_12288_pred.png', '/C_tile_2048_8192_pred.png', '/D_tile_7680_13824_pred.png']

counter=1
for tile_use in tiles_use:

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 16))
    
    F = []
    for k in range(10):
       f = root+'/checkpoints/model1/000'+str(k)+tile_use
       F.append(imread(f)[:,:,0])
       
    for k in range(11,100):
      f = root+'/checkpoints/model1/00'+str(k)+tile_use
      F.append(imread(f)[:,:,0])  
      
    for k in range(101,299):
      f = root+'/checkpoints/model1/0'+str(k)+tile_use
      F.append(imread(f)[:,:,0]) 
        
    img = imread(root+'/Geomorph/val'+tile_use.replace('_pred.png','.jpg'))[:,:,0]    

    lab = imread(root+'/checkpoints/model1/0000'+tile_use.replace('pred','gt'))[:,:,0]

    ax[0].imshow(img, cmap='gray')
    ax[0].imshow(lab, alpha=0.5, cmap='plasma')
    ax[0].set_title("Ground truth", fontsize=10)
    ax[0].set_axis_off()    

    def update(i):
        ax[1].imshow(img, cmap='gray')
        ax[1].imshow(F[i], alpha=0.5, cmap='plasma')
        ax[1].set_title("Epoch "+str(i), fontsize=10)
        ax[1].set_axis_off()

    anim = FuncAnimation(fig, update, frames=np.arange(0, len(F)), interval=200)
    anim.save('model1_evol_valimage_per_epoch'+str(counter)+'.gif', dpi=80, writer='ffmpeg')
    plt.close()
    del fig
    counter += 1
    
# bash crop_gifs.sh    
    
